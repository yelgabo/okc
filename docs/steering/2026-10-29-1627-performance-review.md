# Hourly performance review, 16:27 PDT

Evidence snapshot `f61d8d9`; pending bundle inspected at `b694b4a`. No new benchmark job or
layer score. The [tracker](../architecture-performance-tracker.md) and dashboard now distinguish
an assembled bundle from a layer-tested one.

Family B retains its historical +27.9/+24.2 us loss at 128/256, but its explicit revisit trigger
is met: phase B/setup improved after that comparison. Re-test the staged pair with current
components against current A in one process, using complete-MoE timing, correctness checks
and repeated routing seeds. Keep A until that comparison supports a change.
[Revisit policy and evidence](../ruled-out.md).

The combined A/E bundle with projection updates is assembled at `b694b4a` and remains
unjudged. FIN8 plan assertions and fresh-qkv replay checks are implemented in that branch;
execution is pending. The whole-layer result remains job `9b921febc7ad4140`, 1.964x.
[Bundle and mechanism report](2026-10-29-v1-d32-breakdown.md).

SK_PDL + ONE_TRIG proposes overlap between attention and projection. Recorded default-off
builds match production SASS; enabled builds have compile evidence only. Host launch compilation,
runtime correctness, replay stress and chain timing are still required. Preserve this as layer
context, not a new family D scheduler or family E speedup. The source report defines the bounded
chain test and stopping criterion. No tests were launched by this review.

Checked recorded compile outputs, branch source/log, all four run logs, rejection-register changes,
18 untracked profile JSON job IDs, dashboard filters and source consistency. Profile jobs are
already logged. No independent review of these narrow documentation updates was performed.
Local documentation commit only, as the hourly instructions forbid pushing.
