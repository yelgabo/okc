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

The independent experiments initiated from this supplement are recorded in [the experiment plan](../../superpowers/plans/2026-11-02-cuda-issue-groups.md). A shape-specific WGMMA issue-group treatment is checked against local compilation before GPU allocation. A separate scale-constant treatment questions an inherited instruction-placement trick while retaining the exact mathematical multiplier. Neither has a presumed speedup. D32 Worker's attention confirmation remains separate.
