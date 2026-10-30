# Hourly performance review, 23:35 PDT

Evidence snapshot `dd1777e`. Mode 51's isolated layer gain is now unresolved. The old
three-arm comparison could not detect an incumbent-arm penalty. Four-arm comparison
`2dc566298be1498f` reports 1.0057 pooled against nulls 0.9898/0.9972 and fails its decision
rule. Component gains remain valid. The [tracker](../architecture-performance-tracker.md)
marks historical layer deltas as observations with this limitation.

A/A `724aa0dd6ed140ca` confirms a decode-1 arm offset despite identical code. Runtime M1 + 51
in `40b9b3f795f84073` passes the heuristic at 1.0162 pooled, but includes both mechanisms and
that offset. Approximately 1.2% net after A/A and 0.4% attributed to M1 are cross-run estimates.
All three four-arm runs pass 96 checks; policy variation between copies remains material.
[Arm-bias analysis](2026-10-30-layer-bench-arm-bias.md), [layer runs](../runs-decode-layer.md).

The enforced-gate M1 runtime candidate uses mode 47, not the measured build's mode 51, and
has no check or submission. Keep the 2.014x release. Test the exact candidate with its gates,
and stabilize prepare-time picks before treating small layer differences as resolved.
[Manifest](../release-manifest.md).

Attention diagnostic `2b93c6ecf4f641c6` supports merge-order rounding for the tested mode-49
history cases: deterministic merge is bitwise, production differs by at most a few BF16 ulps,
and workspace scans are clean. The ordering argument now has independent review with stated
assumptions. The separate PDL audit finds no issue on the shipped path, and a latent W4 GEMM2
early exit outside that path is fixed in the unchecked candidate. These do not replace runtime
validation of the exact file. [Attention log](../runs-attention.md),
[ordering review](../review-m1-ordering-2026-10-30.md).

New A-descendant gather mode 57 loses at 32/128/256; mode 59 ties and is not kept. B/C/D
standings otherwise hold. No new isolated E gain is claimed from the combined M1 result.

Principal new results and limitations were independently checked against raw logs. Dashboard
filters, source consistency and local links were checked. The 18 untracked profile JSONs remain
untouched and already map to logged jobs. No benchmarks, kernel edits, submissions or pushes
were performed. Documentation commit is local only.
