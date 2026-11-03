# CUDA part 2: independent compiler and MoE experiments

November 2, 2026. Owner: OKC CUDA Optimizer. User instruction: "Be independent and run some experiments. I want to see positive results." Evidence base `f720ac617d9eb82f3bd69d16be2ff52809f64edd`. [Plan and prospective gates](../superpowers/plans/2026-11-02-cuda-issue-groups.md). [Research correction digest](../research/cuda-h200/part2.md).

## Scope and selected implementation

The supplement's snapshot `779b06ad` and standalone blob `96b5bf91433fc87154de1edc40ce7a31b5c4c025` are verified locally. Its conclusions correct packing, launch-residency and operand-orientation assumptions; it ran no GPU experiment. This optimizer instead tests the selected layer-native MoE body `c7c201ae`, from full control `ba9e327c`, through the already validated standalone four-state adapter.

D32 Worker's attention confirmation remains separate. This work edits no layer candidate, changes no live D32 experiment, and polls none of its jobs. Its private MoE measurements use that problem's separate quota. Public submissions remain outside this experiment.

## Experiment 1: shape-specific issue groups

Hypothesis: four k16 operations per WGMMA commit group for NTT<=4 can reduce collective issue overhead, using the accumulator capacity released by smaller N tiles. Retain two operations for wider tiles and preserve deferred work's completed-k16 boundary. Packing, arithmetic order, stage drains, all proxy fences and barrier counts remain.

Local CUDA 12.8.1 SM90a compilation completes for both bool entries in B4 and MAIN. It rejects the treatment at the prospective no-new-spill gate: B4 spill store/load bytes change from 4/32 to 80/88; MAIN from 0/0 to 4/24. Stop this exact treatment before target allocation. No GPU latency or general issue-group limit follows from this result. [Resource evidence](evidence/2026-11-02-cuda-issue-groups/group-resources.json).

## Experiment 2: constant scale multiplier

The WGS loop computes `(OKC_SC6W ? 4u : 16u) + p.zero`. The immutable host wrapper sets `p.zero = 0`. The original opaque value forces an integer multiply to a particular instruction pipe. Remove that dependency and let the current compiler fold the identical integer multiplier. This preserves reconstructed scale bits, BF16 multiplies, all numeric boundaries, grids, scratch and ordering operations.

The initial local compiled comparison removes B4's spill stores/loads and its eight static LDL plus one STL instructions per entry. Both MAIN entries retain zero spill traffic. HGMMA, wait0, wait1, warpgroup-arrive and async-fence counts match control per entry. This is a positive compiler result, not a measured H200 speedup. [SASS counts](evidence/2026-11-02-cuda-issue-groups/scale-sass.json) and [resources](evidence/2026-11-02-cuda-issue-groups/scale-resources.json) record the observations. The final source removes the stale inline annotation describing the control's opaque multiplier.

## Registered private measurement

[The exact payload](evidence/2026-11-02-cuda-issue-groups/paired.py) and [manifest](evidence/2026-11-02-cuda-issue-groups/paired.manifest.json) preserve source identity and the seven actual compiler requests. Reuse the existing complete harness's three banks, two route distributions, nine correctness sizes, six timed loads, four independent prepared states, fresh capture inputs, occupied/empty transitions, cold replay, reversed traversal, duplicate nulls and cleanup checks. Runtime and decision contract remain byte-identical to their established implementations.

One initial private check is authorized here. Refresh live jobs/quota immediately before it and record the accepted ID. Wait only on that ID. An official score compares the candidate with SGLang; treatment/control improvement comes from the matched capsule. Require all six bank/distribution summaries and four placement crosses to meet the existing >=1% equal-six factor and duplicate-null gate, with no >=1% affected-load regression, before promotion. One further confirmation is available only if the first result is promising. No automatic sweep after a loss.

Current status: local source checks and compilation; private result pending. No timing gain, official correctness pass, target-toolkit resource count or promoted runtime change is claimed at this checkpoint.

## Evidence and remaining gates

Six focused CPU tests verify reversible source edits, group-bank reuse and arithmetic issue order, immutable host suffix, publication text and the scale transform's dependence on the verified host-zero assignment. The [baseline-aware hotspot map](evidence/2026-11-02-cuda-issue-groups/hotspots.md) identifies complete expert execution as the standalone target. Old profiling activity is not treated as exposed removable time.

The compiler driver discloses its host-tail extraction, declaration-only lookup adapter and compile-only references. Actual target compilation, numerical results, fresh replay correctness and latency remain required. The user-authorized experiment does not need a forecast of a win. Independent agent review is unavailable under the no-new-agents directive; no new memory-ordering protocol is being shipped.

## Private check accepted

Private MoE job `3aa3885f6b734f8c` accepted from source commit `ca588999`, payload SHA256 `ccef8a76df0d8bf0e60a882a6161c0d91925ac389743aa70160514b371f2dfd0`, 342733 bytes. The source and seven build descriptors are fixed. Fresh preflight showed 20 MoE checks remaining and only the separately owned D32 confirmation `4fece18b20a2447b` queued. The optimizer owns the sole waiter for its MoE ID and leaves that D32 job with its worker. [Accepted source/job record](evidence/2026-11-02-cuda-issue-groups/accepted.json). Target status and timing remain pending.
