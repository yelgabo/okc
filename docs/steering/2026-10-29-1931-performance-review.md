# Hourly performance review, 19:31 PDT

Evidence snapshot `b41cb8d`. Clean layer release `3fbb3ad027e84c6b`, source `2cee446`, scores
2.014x (95% interval 1.984-2.045), 13/13 passed. Main's layer file matches its SHA256.
The [tracker](../architecture-performance-tracker.md) records new jobs and corrections.

Mode 47 has isolated layer evidence from `720079090ee44cc3`: at 128 tokens, sliding/full
improves 4.9/5.5 us; at 256, 5.7/3.6 us. The whole-score ratio is contaminated by a decode-1
arm artifact and must not be used as an isolated gain. This closes mode 47's pending layer
confirmation, with the limits of one paired check. [Layer report](2026-10-30-layer-release-attribution.md).

Mode 51 replaces it in the pending-layer queue: held-out layer-routing component tests improve
3.9/3.9 us at 128/256 versus 47. The component submission scores 2.202x versus 47's 2.216x,
so the board entry stays 47; cross-run score variation does not cancel paired gains. Mode 50
fails its 32-token performance gate. Tail-split modes 52/53 lose at all tested sizes. B's
previous loss remains parked. [MoE log](../runs-moe.md).

M1's earlier failed replay check is preserved, but its cause is not isolated to M1. The
M1-off control exceeds the same 1e-4 bound; fresh-input on/off trials have matching maxima
and no stale-read signature. M1 remains untimed. Do not change the validation policy or infer
ordering safety from these diagnostics. The first divergent intermediate and proxy argument
are still missing. [Diagnosis and approval boundary](2026-10-30-layer-release-attribution.md).

Next: controlled layer testing of 51 and reconciliation of the 32-token MoE gap. The component
public lead is only 1.5-1.9 us on layer routing, versus the earlier 10-14 us in-layer profile
attribution. Those are different measurements, not interchangeable optimization budgets.
Keep the released baseline and require correctness plus paired gains beyond the null spread.

Principal MoE results were independently checked against raw logs; the tracker/dashboard
consistency pass includes layer attribution and the M1 correction. Local links, filters and
generated source consistency were checked. No benchmark, submission, kernel change or push
was performed. Existing untracked profiles remain untouched. Documentation commit is local.
