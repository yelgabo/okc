# GOPT-WS-03: helper norm-stat aggregation

Actual pickup: 2026-11-03 03:36 UTC by OKC General Optimizer, chat01a18696, after fetching and reading shared steering at `1e068258`. This continues the [ranked refill](2026-11-02-general-hotspot-search-refill.md#1-general-reduce-unit-rate-norm-stat-atomics). No private job is accepted yet.

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

Own job `b38a9ed6dc9743e9`, source revision `51d6286b`, payload `1e857ad4`/344267 bytes, is accepted for muse_moe_w4a16. Fresh shared preflight found17 remaining MoE checks and only Decode32's separately owned layer jobbd5776bdfcab42a7 running. Exact remote payload bytes and regeneration matched before dispatch. [Accepted identity](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/accepted.json), [intent and frozen source](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/intent.json). One owning waiter collects under ignored `.build/gopt-ws-norm/`; archive terminal output only after exit. No correctness or timing result is recorded yet.
