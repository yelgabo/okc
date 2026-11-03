# Hourly performance review, 18:25 PDT

Evidence snapshot `283e348`. The combined A/E and projection package now wins in a
same-process layer comparison. Job `aa4276e91c924dad` measures incumbent/candidate ratio
1.0158 (1.58% speedup, 1.56% less time), two inputs per shape and eight randomized cold-L2
rounds, with 72 checks passing and candidate/null ratio about 1.0006. Release
`edd71d3152a946fe` scores 1.982x (95% interval 1.953-2.012), 13/13 passed.
The [tracker](../../../docs/architecture-performance-tracker.md) records shape deltas and invalidated
attempts. [Layer run log](../../../docs/runs-decode-layer.md).

A mode 37/sk8w3s and E DCARRY/PK16/VHOIST have reached the layer together with qkv8/cgemv.
Keep the combined gain separate from isolated family credit. The corrected FIN8 asserted
mode/replay checks pass. The first degraded-incumbent bench and failed public import supply
no usable comparison; their repaired runs are explicitly identified in the tracker.

B's retest on mode 37 components loses 30.4/24.4 us at 128/256, six of six seeds slower.
Keep it parked until phase B alone falls below about 50 us at 128 or the fused phase A tail
vanishes. Mode 47 improves over 37 by 2.3/3.7 us on held-out layer-routing seeds at 128/256,
and repeats across standalone/skewed routing. It ran in a mixed layer follow-up, so the next
useful test is a controlled layer comparison against the release, not another claim that it
has never reached the layer. [MoE run log](../../../docs/runs-moe.md).

M1 attention-to-projection overlap fails its forced-on replay test at verify 1 full before
timing (hidden 2.0e-4, stream 1.6e-4, bound 1e-4). It was disabled. Resolve the ordering
question and pass replay checks before timing it again. The failure applies to M1, not all
of family E or D. Projection geometry and late-release improvements remain separate context;
their isolated layer effects are unmeasured. [Rejection register](../../../docs/ruled-out.md).

Priority: preserve the winning release; compare mode 47 with it under controlled layer inputs.
Use package subtraction controls only when isolated A/E/projection attribution is needed.
Reject a candidate on correctness failure; require gains beyond the null spread before claiming
an improvement. No kernels, benchmarks or running jobs were changed by this review.

Reviewed the four run logs, raw paired MoE/layer evidence, release receipt and invalidations.
An independent reviewer checked the principal numbers and status distinctions. Dashboard
filters, generated-source consistency and local link targets were checked. The 18 existing
untracked profile JSONs are already logged jobs and remain untouched. Commit locally only;
the hourly automation forbids pushing.
