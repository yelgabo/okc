# Hourly performance review, 14:25 PDT

Evidence snapshot `efddc84`. Family E has new component gains from PK16 probability packing
and VHOIST transpose address/descriptor carry, combined with DCARRY. A-D standings are unchanged.
The [tracker](../../../docs/architecture-performance-tracker.md) records timings and limits.

Same-process check `025ccaf9c57a4ba2` compares the combined switches against DCARRY at
128/256 tokens, full attention. Cold times improve by 5.35/9.63 us (4.6%/4.7%);
ABCHECKSUM is 19/0 for every arm. Candidate `c37d381`, job `569b233dfe08414f`, passes 16/16
at 2.449x (95% interval 2.413-2.502). Repeat `e9085c86d89449c2`, source `61e3fd4`, passes
16/16 at 2.504x (2.467-2.542). Unchanged modes also ran faster on the repeat's machine,
so its score increase does not establish another code improvement. [Attention log](../../../docs/runs-attention.md).

The attention package, mode 37 and sk8w3s still need layer integration evidence. Whole-layer
score remains 1.964x, job `9b921febc7ad4140`, 13/13 passes. Projection changes remain separate
context. Component gains must not be added or credited as layer gains.

Before relying on the next combined layer comparison, resolve the
[community audit's validation gaps](2026-10-29-community-audit.md): assert mode 11 with eight
copies in targeted FIN8 replay, enforce reference acceptance on every layer seed, balance
arm order and isolate mutable cache state. These are coverage gaps, not observed correctness
failures of the incumbent. Keep the production fence; the no-fence probe is diagnostic only.
After those repairs, compare the coherent integrated package with the preserved layer control.
Retain the control if correctness fails or repeat layer timings do not establish a gain.

Reviewed committed reports and raw paired evidence, preserving existing untracked profiles.
The tracker, JSON summaries and generated dashboard were checked for consistency and filters.
No benchmark, port or submission was launched. This review is committed locally; the hourly
automation explicitly forbids pushing.
