# Part 2 corrections and experimental use

The user's November 2 supplement is preserved in [part2-source.txt](part2-source.txt). It is a correction to the first research pass, rather than a new measured performance result. Its repository checkpoint is `779b06ad6b5a1eefcd0a0006b6870668b987f2de`; its MoE blob pin `96b5bf91433fc87154de1edc40ce7a31b5c4c025` matches that Git object locally.

## Corrections to carry into experiments

- Consumer-order packing already exists. A new packing candidate must identify a remaining load, permutation, scale operation or layout conflict in the actual representation.
- Register capacity and launched work are separate limits. The inspected launchers use an SM-count grid. A lower register count alone cannot provide two CTAs per SM throughout the device. Shared memory and the global progress protocol also constrain a larger grid.
- Three- and four-slot skinny rings already have a recorded comparison. The cited `75bca9775f6f4b0b` result must be read through its owning receipt before another sweep. The supplement itself read source annotations, rather than independently reprocessing those results.
- Trace operand orientation before assigning padding. Weights occupy register A; tokens map to the generated N specialization. Instruction M64 is not evidence of a 64/token_count token-padding ratio.
- Proxy fences already appear on specific publication paths. Trace writer, publication, async reader and reuse. Their presence is neither a missing optimization nor a complete correctness proof, and this supplement authorizes no fence deletion.

The inspected standalone source and the scored layer's selected native implementation are different artifacts. This supplement does not complete a new projection, attention or whole-layer audit, certify a selected binary, or establish which prepare-time variant won.

## How this changes the optimizer's work

Keep [the application ledger](application.md), [pipelines](pipelines.md) and [scheduling](scheduling.md) as starting references. Use a concrete instruction/dependency experiment against the selected native source. Preserve the existing packing, launch work and numeric boundaries unless the treatment explicitly changes and validates them.

The independent experiments are recorded in [the experiment report](../../steering/2026-11-02-cuda-part2-experiments.md) and [prospective plan](../../superpowers/plans/2026-11-02-cuda-issue-groups.md). Constant scales remove local compiler spills but fail the complete matched H200 speed gate in own private3aa3885f6b734f8c. An ordinary stable-pointer load produces identical SASS and receives no GPU run. The first grouping prototype skipped a deferred flush because its condition was unreachable at the selected source's ten-operation threshold; it never reached GPU. The corrected scale/group combination rounds to the next reachable boundary, preserves publication operations, compiles without spills and passes all correctness/replay checks in own private3c45788127e945c4 but fails the matched speed gate: all24 whole-six factors0.995560..1.008180. Both exact treatments stop. No runtime promotion or target speedup is established. D32 Worker's attention work and the General Optimizer's separate ring-depth task remain with their owners.
