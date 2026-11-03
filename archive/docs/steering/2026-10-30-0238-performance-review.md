# Hourly review: cold pipeline losses and standalone router gains

Evidence snapshot `8838272`. Reviewed 2026-10-30 02:38 PDT.

Attention roadmap labels A1/A2 are experiments within family E, not family A (MoE).
The earlier DCARRY/PK16/VHOIST wins and their combined layer integration remain valid.
The new cross-tile designs lose on the cold-cache workload and have no layer integration.
A-D and the 128/256-token A-versus-B decision are unchanged. No new whole-layer result
appears in this checkpoint. [Attention evidence](../../../docs/runs-attention.md#standalone-round-10-30-a0-scorecard-and-a1-cross-tile-overlap-in-mode-51).

### E: real overlap tested, cold-cache gate failed

`5575e7d15d3e49a8`, kernel `86f2bec`, generator `8a0a4a0`: attention mode 51 with
`ONE_XPIPE=2` issues P.V before the next stage wait and overlaps Q.K/V transpose. The
control is the board attention code, not MoE mode 51. Each arm passes ABCHECKSUM 19/0;
judge receipt passes 16/16. The bench uses 120 randomized rounds and two held-out inputs
(seeds 1000/1001), with independent control states. Full attention verify 16/32 corresponds
to 128/256 tokens; routing distribution is not applicable.

| Candidate vs control | 128 tokens | 256 tokens | Status |
|---|---|---|---|
| XPIPE=2, cold pooled medians | 117.31 vs 113.41 us: +3.90 us (+3.44%) | 193.25 vs 192.45 us: +0.80 us (+0.42%) | Measured loss; 128-token gate fails |
| XPIPE=2, cold per-input deltas | +3.32 / +4.55 us | +0.54 / +0.99 us | Both inputs slower |
| XPIPE=2, warm per-input deltas | -1.8 / -1.9 us | -3.5 / -3.3 us | Warm-cache gain only; does not pass cold gate |
| Mode 115, one CTA/SM vs mode 51, cold | 137.79 vs 110.27: +27.52 us (+24.96%) | 244.93 vs 187.26: +57.67 us (+30.80%) | Measured loss |
| Mode 115 + XPIPE=1 vs mode 51, cold | 138.37 vs 110.27: +28.10 us (+25.48%) | 239.87 vs 187.26: +52.61 us (+28.09%) | Measured loss |

The first row subtracts pooled medians, not medians of paired differences. Logged pooled duplicate
nulls are -0.38/-0.06 us; no confidence interval for these deltas is reported. Warm percentages
are omitted because the summary does not supply the corresponding control times.
[First bench log](../../../muse_attention_decode/profiles/bench_a0.5575e7d15d3e49a8.log).

The mode-115 rows come from `318ae6fc3c3746ce`, source `682aeaa`: two held-out inputs,
120-round default, 16/16 judge tests, ABCHECKSUM 19/0 for real arms. Nulls are +0.06/+0.42 us.
XPIPE=1 saves 5.06 us (-2.07%) inside mode 115 at 256, but leaves it 52.61 us slower than
mode 51. At 128 it adds 0.58 us (+0.42%) inside mode 115. Q.K-first XPIPE=1 at two CTAs/SM
is compile-infeasible in this build (C7512 register serialization); the one-CTA variant compiles
without C75xx but loses residency. These results reject the tested configurations, not all
cross-tile pipelines. [Second bench log](../../../muse_attention_decode/profiles/bench_b.318ae6fc3c3746ce.log).

### Attention diagnostics and next target

`a92aa9c43f064c49` (`df4060d`) tests one-allocation workspace carving: 16/16 and ABCHECKSUM
19/0, but the control remains 0.54 us slower at decode-1 full on both inputs. Carving fails
its +-0.2 us gate. `7144310358564fa3` (`a0a78ce`, generator locally edited) passes 16/16
judged tests but has no usable bench data: undefined `_AB_CAP` breaks every bench row.
It establishes no capture-order result. Fixed repeat `b7762d764d474eae` (`46df563`), six
held-out inputs, 16/16 and ABCHECKSUM 11/0, finds a 0.5-0.7 us spread tied to workspace state
rather than capture order. L2 placement is the proposed mechanism, not a measured cause.
Cross-state decode-1 full differences below 0.7 us are inconclusive under this evidence.
[Carve log](../../../muse_attention_decode/profiles/bench_c.a92aa9c43f064c49.log),
[failed bench](../../../muse_attention_decode/profiles/bench_d.7144310358564fa3.log),
[fixed repeat](../../../muse_attention_decode/profiles/bench_d2.b7762d764d474eae.log).

The standalone scorecard puts most of the remaining gap in short/sliding shapes. Diagnostic
phase exits in `318ae6fc3c3746ce` assign about 2.4-4.0 us to merge/finalize on decode-1 sliding,
decode-1 full, decode-16 sliding and verify-1 full. These are diagnostic budgets, not achievable
savings. Attention A3 reduction ownership and prepare-time allocation selection remain untested;
A3 is not evidence for MoE family C. Next decisive test is a fixed-state, null-controlled
short-shape reduction comparison, with the log's gate of 0.5 us at decode-1 sliding or 1 us
at decode-16 sliding/decode-1 full. Reopen the one-CTA design only with a resource model that
recovers its residency cost. [Scoped failures and triggers](../../../docs/ruled-out.md).

### Separate projection context: rt and KLUE router gains

Five new receipts pass 9/9. R2a `a91f2152d8974458` (`113d513`) scores 1.505
(1.482-1.527), R2b `ba85985b8c5e4969` (`01ad9d9`) 1.484 (1.461-1.506), and
R2c `6cc72e840403402b` (`b50dcb6`) 1.499 (1.477-1.522). Confirmations
`f2373dedd4f44648` and `b80b817c67d043ad`, both `1e12506` with a header-only difference,
score 1.484 (1.462-1.506) and 1.508 (1.485-1.530). Their spread is not a controlled gain;
the run log attributes most of it to unchanged verify-1 shapes. The prior P1 submission
`1d277e7fe9c44712` at `d0dc80f` is logged at 1.482 (1.460-1.504), but its receipt is absent
locally, as are `44fa08c6d2f44b44` and `6cbd8b981a5d4c3f`.
[Projection round and receipt identifiers](../../../docs/runs.md#2026-10-30-round-2-router-attribution-rt-stable-cublaslt-picks-branch-worktree-agent-a7d5e0782cac00376).

R2c records 100 seeded cold rounds, paired against the pre-round production router picks:

| Variant / tokens | Paired delta | Control median | Delta / control median |
|---|---:|---:|---:|
| rt8w8 / 8 | -1.25 us | 5.38 us | -23.23% |
| rt16w8p8 / 16 | -0.88 us | 5.33 us | -16.51% |
| rt16w8p8 / 32 | -0.40 us | 5.74 us | -6.97% |
| wd KLUE / 128 | -0.80 us | 7.63 us | -10.48% |
| wd KLUE / 256 | -0.74 us | 9.97 us | -7.42% |

Percentages use paired deltas divided by control medians after the logged 4.24 us replay
overhead subtraction. They are component results, not layer savings. Nulls are about
0.02-0.13 us for rt and -0.13/-0.06 us for KLUE at 128/256; no paired confidence intervals
are recorded. FP64 checks cover two router matrices, outliers/tiny values, adjacent sizes and
fresh-input graph replays. Routing distribution is not applicable to these projection operators.
[R2c log](../../../muse_bf16_proj/profiles/bench_r2c.6cc72e840403402b.log).

Push reduction is the strongest single tested wd change (-0.46 to -0.66 us across R2a/b).
KLUE combines packing, promotion, push reduction and prologue changes; do not assign its full
win to one ingredient. Removing the producer warp with its barrier loses +1.74/+1.25 us at
128/256; XA orientation is a null, and rt8w8 loses +1.50 us at 32 in R2c. Stable Lt choices
now agree across confirmation preparations except o128 at the margin. P3 compressed WGMMA
remains unbuilt and parked for budget, with a raw-BF16 twin required before testing decode.
[Projection evidence](../../../docs/runs.md), [scoped losses](../../../docs/ruled-out.md).

The [port guide](../../../docs/proj-port-guide.md) still records unresolved outgoing `st.async` sender-exit
and rt self-rank questions. Passing correctness checks does not resolve them; TAILSYNC is the
recorded fallback, costing about 0.1-0.2 us for rt and 0.3 us for wd. No projection layer port
is present. These wins therefore stay separate from A-E and from layer promotion.


The [tracker](../../../docs/architecture-performance-tracker.md) preserves prior results and the updated
comparison ledger. No GPU jobs, kernel edits, submissions or pushes by this review.

Validation: independent source review checked the ten new receipts, timing claims,
percentages and local links. The generated tracker fingerprint matches; executable filter
checks pass for all five filters and 22 ledger rows. E appears under both wins and losses.
No layout changes were made. Browser screenshot verification was unavailable: the browser
wrapper returned an undefined-page-id error and produced no screenshot file.
