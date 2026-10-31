# Hourly review: designated finisher tested; ownership stays ahead

Reviewed 2026-10-30 14:05 PDT. Evidence snapshot `bc10cb1`.

Corrected mode 74 at `4187b8b` loses against mode 51 in job `e02f60fec94f45cb`.
This is a stream-K alternative to C inside A, retaining D-related phase-B claims.
It uses a designated finisher instead of mode 72's last-arrival fixup. The finisher is
chosen during setup from the cost walk. It is now tested, rather than an untested idea.
[Design and gate](../runs-moe.md#designated-finisher-for-stream-k-phase-a-modes-74-and-75-2026-10-30).

Six seeds 107-112 per size/routing use cold-L2 graph replay, medians of 20 replays per arm.
The following are means of paired component deltas, not whole-layer results. Percentages
use the logged control means; rounding can differ from subtracting the displayed arm means.
No paired confidence interval is recorded.

| Tokens / routing | Mode 51 control | Mode 74 delta | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 80.3 us | +7.9 us | +9.84% | 0/6 |
| 32 / standalone | 143.6 us | +7.6 us | +5.29% | 0/6 |
| 128 / layer-like | 163.2 us | +4.6 us | +2.82% | 0/6 |
| 128 / standalone | 201.4 us | +7.4 us | +3.67% | 1/6 |
| 256 / layer-like | 219.9 us | +2.0 us | +0.91% | 1/6 |
| 256 / standalone | 233.1 us | +4.1 us | +1.76% | 1/6 |

The receipt passes, 2.1974x with 95% interval 2.1645-2.2304. Its judged path is the
unchanged board dispatch, so this score is not mode 74's standalone score. First-seed
output checks against 51 are 1.3e-4 to 3.4e-4 relative L2; spin flags are zero.
The diagnostic requires a negative mean at 256 on both routings. It fails, so no held-out
or skewed gate ran and mode 74 remains parked.
[Raw log](../../muse_moe_w4a16/profiles/bench_sk74fix.e02f60fec94f45cb.log),
[receipt](../../muse_moe_w4a16/profiles/bench_sk74fix.e02f60fec94f45cb.json).

The earlier build `5f35018`, job `eaa881a934aa471e`, used slow SiLU because its predicate
excluded modes 74/75 and used serial finisher designation. Preserve its results as that
build's loss, not the corrected design's result. Six seeds 95-100, same replay method:

| Tokens / routing | Mode 51 control | Earlier mode 74 delta | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 85.1 us | +8.2 us | +9.64% | 0/6 |
| 32 / standalone | 142.1 us | +9.0 us | +6.33% | 0/6 |
| 128 / layer-like | 159.9 us | +8.3 us | +5.19% | 0/6 |
| 128 / standalone | 209.3 us | +7.2 us | +3.44% | 0/6 |
| 256 / layer-like | 222.0 us | +2.1 us | +0.95% | 0/6 |
| 256 / standalone | 248.9 us | +3.8 us | +1.53% | 1/6 |

That receipt passes, 2.2023x with interval 2.1693-2.2354. First-seed output differences
are 1.5e-4 to 2.0e-4, spin flags zero. No paired interval is recorded.
[Earlier log](../../muse_moe_w4a16/profiles/bench_sk74.eaa881a934aa471e.log),
[earlier receipt](../../muse_moe_w4a16/profiles/bench_sk74.eaa881a934aa471e.json).

The corrected stamped twin reduces phase-A end spread to 6.6-8.3 us at 128/256,
versus 10.9-19.5 for the control. Fixups cost 3.9-4.5 us per CTA, but route plus setup
still costs 3.4-3.6 us more. Mode 72's larger fixup cost and the public kernel's similar
end spread come from other processes/seeds. They are diagnostic comparisons, not paired
speedups. The remaining setup cause is unresolved. Reopen this implementation only with
setup under 1 us above 51, per-share fixup under 1 us, or a refitted cost table, then apply
the paired gate. These findings retain C for this comparison, not reject stream-K generally.
[Measured stamps and hypotheses](../runs-moe.md), [revisit triggers](../ruled-out.md).

### Separate projection context: one-wave QKV cluster split loses

Job `ad6c673d9a444c1e`, source `90ee4b9`, tests qkv8c4/qkv16c4/qkv32c4 and wider
c4w variants, 72 rows per CTA and four CTAs per cluster. All configurations passed FP64,
2-ulp twin checks and 22 identical one-hot columns. The dedicated paired timing loop failed
with a `torch.sum(out=)` error. These results are prepare-time tuner observations only:
cold-L2 graph replay, 24 alternating repetitions, one preparation, overhead included.

| Tokens | Unsplit qkv reference us | c4 us / delta / percent | c4w us / delta / percent |
|---|---:|---|---|
| 1 | 13.8 | 23.0 / +9.2 / +66.67% | 24.9 / +11.1 / +80.43% |
| 8 | 13.9 | 23.7 / +9.8 / +70.50% | 25.5 / +11.6 / +83.45% |
| 16 | 14.7 | 24.1 / +9.4 / +63.95% | 25.7 / +11.0 / +74.83% |
| 32 | 15.6 | 27.9 / +12.3 / +78.85% | Not tested |

At one token the tuner selects cgemv at 12.8 us; the table uses qkv8 at 13.8 us
as the unsplit geometry reference, not the selected incumbent.
No paired confidence interval exists, and 128/256 were not tested by this experiment.
Raw twins also lose. Receipt passes 9 tests, 1.5089x with interval 1.4863-1.5315, but the
judge selected production picks, not c4. The experiment is parked; production is unchanged.
Cluster residency and decode geometry are hypotheses for the loss, not established causes.
Reopen with measured active clusters at least 32, or a configuration at most 113 KB shared
memory whose raw twin beats qkv16raw.
[Run record](../runs.md), [raw log](../../muse_bf16_proj/profiles/proj_c4_bench.ad6c673d9a444c1e.log),
[receipt](../../muse_bf16_proj/profiles/proj_c4_bench.ad6c673d9a444c1e.json), [scoped rejection](../ruled-out.md).

### Family F registration and next decision

Family F is now registered separately as the donor-based baseline/hybrid, pinned to
alygg77 source `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`. F0 is the exact donor,
F1 a documented conforming adaptation, F2 a measured hybrid. Projection, attention, MoE
and layer tracks are pending. No controlled F gain, correctness clearance or promotion is
established. Archived donor scores are external evidence only.
[Registration](2026-10-30-family-f-donor-baseline.md).

The shared decision now prioritizes complete donor-versus-release projection and attention
comparisons before new GPU rewrites, with independent preparations and null controls.
A coherent replacement or measured hybrid is an allowed outcome. A-versus-B at 128/256
remains unchanged: retain the working baseline, and reopen staged B only under its stated
phase-B/phase-A-tail trigger. E's PRQ gain and XPIPE losses stand; layer_hp and revised
mode-51 layer candidates remain pending. Whole-layer release stays 2.014x. No gains are
added across families or projection context.
[Donor-first decision](2026-10-30-donor-first-adoption.md), [layer log](../runs-decode-layer.md).


Validation: independent source review checked the three receipts, timing arithmetic, attribution and F registration. Its QKV reference-label and Markdown table findings were fixed and rechecked. Local links, six-family filters, 25 comparisons and five pending cards passed checks. The regenerated tracker fingerprint matches. Chrome inspection of the F card and pending section found no layout collisions. All 18 untracked receipts remain historical and already logged; none were staged. No benchmarks, submissions or pushes were performed.
