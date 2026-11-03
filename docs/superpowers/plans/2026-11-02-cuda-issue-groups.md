# Shape-specific WGMMA issue groups

Owner: OKC CUDA Optimizer. November 2, 2026. Source checkpoint `f720ac617d9eb82f3bd69d16be2ff52809f64edd`. The user's new instruction authorizes independent experiments. D32 Worker's attention confirmation stays separate.

## Hypothesis and alternatives

The selected native WGS loop commits two k16 instructions per group for every N specialization. Its comment attributes this to the wide specialization's register budget. A smaller token tile has fewer accumulator registers. Test four instructions per group for NTT<=4 and retain two for larger tiles. The added operand registers should fit the accumulator capacity released by a smaller NTT. This can reduce collective fence/commit/wait overhead while preserving arithmetic order and operand lifetimes. Benefit remains unknown.

The complete alternative is to keep the existing decoder, packing, rings, scheduler and epilogues while changing group granularity by NTT. A broad replacement engine would add unrelated format/normalization work. A register-cap sweep cannot create unlaunched blocks. A ring sweep repeats known experiments. These alternatives do not address the specific collective-instruction overhead as directly.

## Exact treatment

Use immutable layer control `ba9e327c`, native `c7c201ae`, in the existing standalone MoE adapter. Change the WGS `GS` constant. The original intent to preserve the exact deferred-work boundary was incompatible with the selected FCI=4/GS=2 threshold of ten. The corrected treatment below rounds to the next group boundary. Keep the grid, CTA size, shared-memory layout, all proxy fences, stage drain, barrier counts, numerical boundaries and dispatcher thresholds. No production source is edited during exploration.

Two operand banks alternate by committed group. After committing group g, wait_group1 completes all groups older than g. Before bank g mod2 is reused for g+2, its previous reader g has therefore completed. Keep the stage's wait_group0 before activation-slot release. The fence preceding each group's register operands covers all four instructions in that group. This changes grouping rather than removing required publication. See the CUDA 12.8.2 PTX ISA's WGMMA fence, commit_group and wait_group sections.

## Gates and budget

1. Generate exact reversible edits, check sixteen k16 instructions and their accumulation order for NTT1..8, and verify the unchanged host/synchronization text.
2. Compile control and treatment for actual B4/MAIN defines on local CUDA 12.8.1. Reject new spills, lost WGMMA pipelining or changed launch bounds before using GPU quota. Compiler results do not establish target correctness or speed.
3. Reuse the existing complete standalone four-state harness, legal routes, cold captured replay, duplicate controls, held-out banks and strict numeric checks. Register its source and selected compiler descriptors. One initial private MoE check, then at most one confirmation if promising. No scored submission.
4. Keep the existing >=1% equal-six factor and duplicate-null gate as the strong promotion threshold. Report smaller/inconsistent changes as such; never turn an official versus-SGLang score into a treatment/control gain. Stop an exact losing treatment and record it with its scope. After three weak experiments, reframe rather than extending a knob sweep.

Complete the supplement's reference update and publish the source, job, observed results, next decision and unresolved limits in the steering index. Preserve other sessions' files and accepted jobs.

## First compiler result and second treatment

The exact GS4-at-NTT<=4 treatment fails the local no-new-spill gate. B4 changes from 4/32 spill store/load bytes to 80/88; MAIN changes from 0/0 to 4/24. Stop this exact candidate before GPU allocation. These are compiler results, not GPU latency.

The second bounded treatment removes `+ p.zero` from WGS scale reconstruction. Its immutable host wrapper assigns `p.zero = 0`. The original opaque value forces an IMAD rather than constant-folded integer arithmetic. Make the identical multiplier constant, retain all scale bit patterns and rounding, and let the compiler schedule the result. This tests whether the old instruction-pipe placement still helps the selected SC6 implementation. It changes neither resource capacity nor launched work. Apply the same compiler and matched replay gates, using the existing harness rather than a new measurement framework.

## Concurrent compiler-only interaction test

While private scale job `3aa3885f6b734f8c` runs, test scale folding together with replacing WGS's repeated volatile shared load of `S.ssp` by the existing ordinary-pointer form. The old source says hoisting creates spills. The new scale-only code removes B4 spills, which is a concrete changed register-lifetime premise. `S.ssp` has one assignment per launch, before routing and the CTA/join barriers, and remains fixed while consumers run; these barriers and every fence stay. This is a compiler-only discriminator, not another GPU allocation or a change to the accepted source. Stop if it adds spills or changes collective counts. No claim about its performance is made.

## Reframe after the first matched result

Private `3aa3885f6b734f8c` passes all official and private correctness/replay checks. Its 24 whole-six factors range from 0.992411 to 1.004932 and all fail the frozen gate. A descriptive aggregate is 0.997953; unchanged short paths also drift, so this is not a causal 0.2% slowdown. The affected-load mean factors are approximately 1.000066, 0.999202 and 1.001283 at T32/T128/T256. Stop the scale-only treatment as a performance candidate.

The three small experiments identify the limitation of instruction cleanup: group-only batching adds spills, pointer spelling produces identical SASS, and removing scale-induced spills leaves most matrix loops unchanged and no measurable endpoint gain. This does not establish a hardware ceiling. The assumptions were the current two-step grouping, source register budget and existing packing/scheduler.

Give the strongest interacting alternative a bounded test: combine constant scales with four-step groups at NTT<=4, preserving the existing two-step wide cases. The scale change has demonstrated a different allocation outcome, and may let batching reduce collective matrix-loop overhead without the group-only spills. Neither ingredient must win independently. First compile both selected candidate builds against the already compiled control. Stop at new spills or lost pipeline invariants; otherwise one new private comparison under the same complete-call gates is within the user's independent-experiment request. Do not extend this into a threshold/ring/register-cap sweep.

## Schedule audit correction

The initial equality test skipped the first-stage flush for four-step groups, because the active native threshold is ten, not the nearby helper's eight. Neither grouped source reached GPU allocation. Those compiler results describe the flawed exact sources and do not reject correctly scheduled grouping. The accepted scale-only GPU source is separate and unchanged.

The corrected combination uses `ceil(10/GS)-1` as the zero-based flush group. This flushes at twelve for GS4 and ten for GS2. The extra work lies inside an already acquired stage, before any next-stage ring wait, so it adds no producer dependency on the delayed release. The phase-end flush and all publication operations remain. Test active constants and all possible original group boundaries, then inspect SASS publication counts and spills. Preserve initial artifacts and write corrected artifacts to a separate directory.

## Corrected combination private registration

Local CUDA12.8.1 B4 and MAIN both pass with 96 registers and zero stack/spills for both bool entries. Per-entry HGMMA512, wait0=48 and async-fence84 match control. Wait1 and warpgroup-arrive decrease256 to192, a distinct matrix-loop treatment. These are static compiler observations, not a speed forecast. Test native `7db5ec77` versus `c7c201ae` through the same complete four-state harness and immutable matched gates. One initial private MoE check; one confirmation only after a promising result. Preserve all six timed loads, both B4/MAIN alternatives, and duplicate-null/regression gates. No public action or D32 integration is authorized by this registration.

## Terminal decision

Corrected combined job3c45788127e945c4 passes official5/6/6 and every private correctness/replay check. All24 whole-six factors0.995560..1.008180 fail the frozen gate; descriptive mean1.001063 is within control spread. Three affected bank/load cells regress>=1%. No confirmation or promotion. The scale-only and corrected combined sources stop; the unchanged baseline remains selected.

Keep the positive local compiler observations scoped: zero spills and25% fewer static group waits/arrivals are not an endpoint gain. Both own pollers are terminal. Full receipts, profiles, strict analyses, final22 CPU checks and current decisions are in the experiment report. The next discriminator is phase timing with an observer-cost control, rather than another automatic register/group/ring sweep. No new job or ownership assignment is made by this plan closure.
