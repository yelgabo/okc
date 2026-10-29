# Hourly performance review, 13:27 PDT

Evidence snapshot `615539c`. New work is projection-only: sources `803178f` / `a35709c`,
runs `5ff76a62b0044378`, `f6c80b4ba11048f3`, `81254e8aed794eab`, reported in `754549d`.
The [tracker](../architecture-performance-tracker.md) records the paired timings and limits.
The dashboard now identifies this as context outside families A-E.

Single-token qkv_gate improves 0.45-0.54 us, pooler 0.86-0.96 us and unpooler slice 0.22-0.35 us
against the previous file in two same-process, cold-L2 passes of 300 randomized rounds each.
The o projection is inconclusive. ABCHECK passes. Projection score 1.444x has a 95% interval
of 1.422-1.466; it is not a whole-layer score. [Run log](../runs.md).

A-E standings and their three pending layer tests are unchanged. The new cgemv and earlier
qkv8 also need layer evidence. The [projection log](../runs.md) supplies port and correctness
checks; verify actual layer timings before crediting a layer gain. No port or benchmark was
performed by this review.

The wide concat proposal at 128 tokens was declined analytically in `f88e278`. Its idealized
24.4 us estimate is above the measured cuBLAS control's 24.0 us. Preserve this as an analytical
rejection, not a measured loss of an implemented challenger. It is unrelated to family B's
staged MoE result. [Rejection register](../ruled-out.md).

Fetched origin/main; no remote-only commits at review time. Checked source reports and the
paired raw log, not an independent recomputation of all samples. Existing untracked profiles
were preserved. Local documentation commit only: the automation forbids pushing.
