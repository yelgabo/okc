# Shape-specific WGMMA issue groups

Owner: OKC CUDA Optimizer. November 2, 2026. Source checkpoint `f720ac617d9eb82f3bd69d16be2ff52809f64edd`. The user's new instruction authorizes independent experiments. D32 Worker's attention confirmation stays separate.

## Hypothesis and alternatives

The selected native WGS loop commits two k16 instructions per group for every N specialization. Its comment attributes this to the wide specialization's register budget. A smaller token tile has fewer accumulator registers. Test four instructions per group for NTT<=4 and retain two for larger tiles. The added operand registers should fit the accumulator capacity released by a smaller NTT. This can reduce collective fence/commit/wait overhead while preserving arithmetic order and operand lifetimes. Benefit remains unknown.

The complete alternative is to keep the existing decoder, packing, rings, scheduler and epilogues while changing group granularity by NTT. A broad replacement engine would add unrelated format/normalization work. A register-cap sweep cannot create unlaunched blocks. A ring sweep repeats known experiments. These alternatives do not address the specific collective-instruction overhead as directly.

## Exact treatment

Use immutable layer control `ba9e327c`, native `c7c201ae`, in the existing standalone MoE adapter. Change the WGS `GS` constant and preserve deferred work's issue boundary at the same completed k16 count. Keep the grid, CTA size, shared-memory layout, all proxy fences, stage drain, barrier counts, numerical boundaries and dispatcher thresholds. No production source is edited during exploration.

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
