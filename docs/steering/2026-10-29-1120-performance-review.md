# Hourly performance review, 11:20 PDT

Evidence snapshot `14d6da1`; new evidence is run `ca43908f02874886`, code `3f67a45`,
reported in `eced63d`. Reviewed the latest steering report and changes since `d0c062b`.
The [performance tracker](../architecture-performance-tracker.md) and visual dashboard now
include this correction. This report adds no new experiment recommendation.

## Change in evidence

Corrected timers invalidate the earlier claim that fat MoE consumers never wait. Weights and
activation waits occupy about 20% of consumer stage-loop time at 128 and 256 tokens. The
instrumented build adds 7.0 / 6.1 us versus production mode 37 in six paired layer-routing
seeds. Production instruction text is unchanged according to the SASS report. The run passes
5/5 tests. This is corrected attribution, not a production performance loss.

Sources: [raw log](../../muse_moe_w4a16/profiles/bench_fw1.ca43908f02874886.log),
[run report](../runs-moe.md#fat-path-waits-with-the-first-try-timed-2026-10-29),
[mechanism proposal and stopping criteria](2026-10-29-fat-moe-waits.md).

## Family status and next evidence

A and C retain their component wins, and dynamic phase B claiming remains a partial realization
of D. B's staged pair remains a measured loss. E's DCARRY improvement remains a component
result awaiting layer evidence. No new layer outcome appears in this review.

The integration gaps remain mode 37, sk8w3s and ONE_DCARRY. The earlier steering report's
per-group load/hoist proposal is still a hypothesis. Preserve its bounded test and stopping
criteria; no implementation or benchmark was launched by this review. Per-warp waits cannot
be added into predicted whole-layer savings.

## Verification and limits

Checked raw timing rows, report arithmetic and dashboard metadata. No independent raw-sample
recomputation or memory-order proof audit. Fetched origin/main and found no remote-only commits
at review time. Existing untracked profiles were left untouched. This report is committed
locally; the hourly automation's explicit no-push constraint overrides the general publishing
rule. No claim of a remote handoff is made.
