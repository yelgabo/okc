# Hourly performance review, 00:32 PDT

Evidence snapshot `a48edef`. Two new scored runs read 2.013x, within the current release's
range. `b68bdcd7a1ba464c` lost all SUMMARY lines, so M1 status is unknown. In
`d6c30f88c617438b`, M1's gate fails at 1x1 sliding and disables it for the entire run.
Both pass 13/13 judge tests. Neither establishes an enabled-M1 performance gain.
[Tracker](../../../docs/architecture-performance-tracker.md), [layer log](../../../docs/runs-decode-layer.md).

The gate failure occurs where M1 changes no launch. Rare rounding noise is a hypothesis
awaiting the prepared reproduction. Two route sensitivity failures came from zero-row
injections that changed nothing; the tool fix has CPU evidence, not a new GPU result.
The run log records a decision to gate only shapes with actual M1 launches, failing closed
when the launch census fails. This review changes no policy.

Revised layer_m1rt51c and layer_g51b are unchecked. The stable-tuner build has CPU rule tests
and a prepared A/A comparison, but no GPU measurement. Keep the 2.014x release and the
inconclusive mode-51 layer classification until controlled comparisons resolve the effect.
Check the exact revised files and inspect the retained gate/policy summaries before crediting
M1 or promoting a candidate. [Manifest](../../../docs/release-manifest.md).

Principal receipts, raw gate status and candidate identities were independently checked.
Dashboard filters, source consistency and local links were checked. Existing untracked profiles
were preserved. No benchmark, kernel edit, submission or push was performed; commit is local.
