# Timed SiLU: startup ABI counterexample and actual instruction budget

November 2, 2026, 10:23 UTC. Evidence base: allocation8f69050113d8f3bb9102f8f7b36c3ec4cb9fa70e; archived original-control resources/PTX1907c4d97d8c4978598c9935f4bb1ff5d17d20c0. Public incumbent remains607eb52bdaf24160, sourceba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. No new challenge job, quota read, compile, driver action, numerical result or performance gain.

## Decision-changing finding before target execution

Worker ordinal16 is authoritatively running (started10:03:33.037UTC), not stuck or a historical intention. Its fixed35minute allocation therefore ends10:38:33UTC. Existing worker <t3-thread://v2/471a5a44-f247-4d6c-b35d-7f45e6f30552/1c3bc6ca-ab58-48bf-bcfe-f7c2de184b25> retains implementation/dispatch ownership; this allocation has NO GPU authorization.

Captain inspected the actual in-progress generator, not merely its planned contracts. At the pinned WIP observation it appends the table only in _fork_native_ma_prepare. The new host binding requires scratch.size()==4. However the actual unchanged release startup independently prepares private scratch by calling scratch_factory(device), with _ma_make_scratch passed as that factory. The actual factory returns three tensors. The production-only append does not reach this independently prepared state: a B4/MAIN startup call would fail the four-entry host check before testing the kernel. This is an implementation integration counterexample, not a GPU failure or rejection of timed tables. The generator may change after the snapshot; it is not a final artifact.

CPU reproduction isolates and executes the actual pinned public607 factory under tensor-allocation stubs, rather than inventing a replacement factory. Production-after-append has4entries; a fresh independent private allocation still has3. It passes under the installed WSL host-validation Python. This tests list/ABI integration only, not CUDA correctness. See [scratch-counterexample.py](../../../docs/steering/evidence/2026-11-02-captain-silu-instruction-budget/scratch-counterexample.py) and [WIP observation](../../../docs/steering/evidence/2026-11-02-captain-silu-instruction-budget/wip-observation.json).

A second exact interface obligation: existing host code assigns optional p.dbg from scratch[3]. Reusing that slot for the table silently gives the same storage two ABI meanings. Debug writes have compile-time guards; this audit does NOT claim they executed or corrupted the table. The treatment must explicitly preserve inactive debug behavior without aliasing an active writer. No additional instrumentation job is warranted.

Captain sent a narrow active-work steer (message8b4e0964-52a2-4c1c-81c0-9e6d9091cd9d) immediately: cover the actual private scratch factory, preserve debug semantics, retain the original deadline. This is the minimum integration needed by the originally authorized single treatment, not a second optimization, shape rescue or renewed readiness cycle.

## What the existing activation actually emits

Independently decompressed original B4 and MAIN selected native/PTX archives; verified nativec7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71 and both manifest PTX hashes. The WGS nonlinear expression at native line1975 has the same five-operation PTX signature throughout each archived bool entry:

- FP32 multiply by raw0xBFB8AA3B (negative log2(e) constant).
- ex2.approx.f32.
- FP32 add1.
- div.approx.f32, with the rounded/widened gate as numerator.
- FP32 multiply by the separately rounded/widened up value.

Each family has108static source-site copies per bool entry. These are compiler-unrolled/inlined copies, NOT108executions per token, independent candidates, SASS instruction counts or timing samples. The adjacent source retains both fork_bf16 boundaries. The table replaces the first FOUR operations, not all five: the final multiply, BF16 output boundary, address calculation, store and publication remain. Consumer indexing/lookup, full timed table generation and launch are replacement costs. PTX div.approx is not a promise of a single hardware instruction; no new SASS/lowered-cycle claim is made.

The reproducible [audit.cjs](../../../docs/steering/evidence/2026-11-02-captain-silu-instruction-budget/audit.cjs) verifies every identified source-site block, entry association, constants and original hashes. Negative parser cases reject a different file ID and line19750. [result.json](../../../docs/steering/evidence/2026-11-02-captain-silu-instruction-budget/result.json) retains exact representative operands and hashes. This reuses existing original-control artifacts; it performs no compiler work or donor re-audit.

## Priority, gates and stopping conditions

Continue ONLY the existing finite source/CPU/secondary-compile task. A final artifact must cover actual production AND independent startup preparation, preserve selected short paths, allocate private mutable table storage and rebuild it in each affected timed invocation/replay. Review the actual corrected integration and negative tests, not just test counts. The allocator-only CPU counterexample is resolved only by inspecting/running the final real-entry contracts.

The unchanged incumbent-relative resource and codegen gates still apply to all four selected treatment entries. These are necessary but not sufficient for a private runtime allocation. No static instruction count proves the earlier credible>=1%whole-score opportunity hurdle. In particular the exact four-op substitution is not an entire-expert removal; the prior conditional-budget scenarios remain conditional, not forecasts. Actual exposed nonlinear latency, dependent-load cost, table producer cost, numerical equivalence and whole-layer timing remain missing.

Stop a concrete contract/resource/codegen failure or10:38:33UTC expiry; publish exact implementation/STOP without a rescue sweep. Do not dispatch a target check automatically, change production/default or reopen G224/RMW/seam. On worker publication captain will independently verify its committed pins and this specific defect before selecting the next allocation. Top100 remains unachieved; this turn's progress is a reproducible pre-target defect caught and redirected, not a performance winner.
