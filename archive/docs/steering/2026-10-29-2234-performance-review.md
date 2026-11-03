# Hourly performance review, 22:34 PDT

Evidence snapshot `6cb0796`. Mode 51 now has paired layer gains against the clean mode-47
release: job `29fd004fd58e40ef` saves 5.1/5.3 us at 128 tokens and 6.5/1.1 us at 256,
sliding/full. Sixteen paired samples per shape, 72 checks pass, aggregate speed ratio 1.0084,
null 0.9997. The [tracker](../../../docs/architecture-performance-tracker.md) records controls and limits.

The candidate is not released. Check `a438b82ff2c141c2` scored 2.022x (1.991-2.052), 13/13,
on the first gate version; the final file was tested comment-stripped in the bench and has no
unpatched scored confirmation. The board release remains mode 47, 2.014x.
[Manifest](../../../docs/release-manifest.md).

M1 now has diagnostic timing. Its same-build improvement at 8 tokens is about 5 us, but
`fdf75ba5a8124544` shows little net benefit over the release at <=32 tokens. Its build changes
projection issue order too, and validation is report-only. Keep that measurement separate
from a validated release improvement. The prefetch probe is inside its null spread.
[Layer runs](../../../docs/runs-decode-layer.md).

The run log records a validation-policy change and a written, independently unreviewed PTX
ordering argument. An attention input-history lead remains untested: fresh-input differences
with bitwise repeated-input tests warrant the prepared history diagnostic. They do not yet
establish a stale read, but they also prevent assigning all variation to atomic merge order.
No validation rule or kernel was changed by this review.

Next: confirm the exact final mode-51 file, run the prepared history diagnostic under the
coordinator's authorization, and independently review the ordering argument. Preserve the
released baseline. B's retest loss, C/D retained mechanisms and tail-split losses are unchanged.
Do not count M1's within-build savings as a release gain.

Principal raw mode-51 timing and M1 distinctions were independently checked. Local links,
dashboard filters and generated source consistency were checked. All 18 untracked profile
JSONs remain untouched and already map to logged jobs. Documentation commit is local only;
the hourly instructions forbid pushing, benchmarks and submissions.
