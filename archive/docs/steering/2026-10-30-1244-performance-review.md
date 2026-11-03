# Hourly review: projection split exit barrier wins paired router tests

Evidence snapshot `c98018f`. Reviewed 2026-10-30 12:44 PDT.

This is separate projection context, with no new A-E or whole-layer result. Job
`4f3ff774981a494c` at `db0f8ad` compares relaxed split exit barriers (wd KLUEYZ,
rt mode 3) with a copy of `ce25f67`'s conforming tail (wd KLUET, rt mode 1).
The arrive moves after the receive wait and before reduction/output stores; the exit wait
stays. Router workloads use 8/16/32/128/256 tokens, not MoE routing distributions.
All arms passed FP64 checks on two router matrices, adjacent sizes and two graph replays.
[Run record](../../../docs/runs.md), [raw paired log](../../../muse_bf16_proj/profiles/proj_tail_bench.4f3ff774981a494c.log).

One same-process experiment uses 100 seeded cold-L2 rounds. Deltas are paired medians;
percentages divide that delta by the logged control median, rather than subtracting marginal
medians. No paired confidence interval is recorded.

| Router tokens | Control median | Relaxed split delta | Percent | Rounds faster | Production-copy delta |
|---|---:|---:|---:|---:|---:|
| 8 | 4.64 us | +0.00 us | +0.00% | 45/100 | -0.06 us (-1.29%) |
| 16 | 4.96 us | -0.22 us | -4.44% | 76/100 | -0.18 us (-3.63%) |
| 32 | 5.79 us | -0.03 us | -0.52% | 51/100 | +0.00 us (+0.00%) |
| 128 | 7.71 us | -0.45 us | -5.84% | 94/100 | -0.32 us (-4.15%) |
| 256 | 10.21 us | -0.32 us | -3.13% | 86/100 | -0.26 us (-2.55%) |

The wd relaxed duplicate saves 0.42/0.29 us at 128/256 (5.45%/2.84%; 88/87 rounds
faster), while duplicate-control nulls are +0.00/-0.03 us. The rt 8/32 results remain
inconclusive. The release-arrive split loses +0.35/+0.26/+0.40 us at 8/16/32
(+7.54%/+5.24%/+6.91%; 7/16/9 rounds faster); its wd savings are only 0.14/0.16 us
(1.82%/1.57%; 71/72 rounds faster). The no-exit-barrier arm is a diagnostic bound,
not a candidate. [Scoped rejection](../../../docs/ruled-out.md).

The bench receipt passes 9 tests and scores 1.4972x (95% CI 1.4747-1.5196).
Clean check `25ea2b62ff8a47a7` at `2265fa2` passes 9 tests, 1.4931x
(1.4707-1.5155). Coordinator scored run `d85d0081fa8a4a43`, same source, passes
and scores 1.4976x (1.4751-1.5200). These overlap the previous 1.495/1.496 runs:
aggregate improvement remains inconclusive across runs. The paired gain is relative to
`ce25f67`, not proof of a gain over the older projection baseline at 256.
[Bench receipt](../../../muse_bf16_proj/profiles/proj_tail_bench.4f3ff774981a494c.json),
[clean receipt](../../../muse_bf16_proj/profiles/submission_tail1.25ea2b62ff8a47a7.json),
[scored receipt](../../../muse_bf16_proj/profiles/scored.d85d0081fa8a4a43.json).

The recorded ordering review calls the split sound with notes. Its relaxed form relies on
arrive executing after the receive-wait loop; the guide notes that the ISA does not explicitly
state the barrier-speculation property. Passing tests do not establish ordering safety.
A future CLR>1 PUSH configuration needs a fresh review. The o128 library pin is still
unresolved because six preparations select different winning configurations; P3 stays parked.
[Ordering argument and caveats](../../../docs/proj-port-guide.md), [run record](../../../docs/runs.md).

Keep standalone promotion separate from layer integration. Projection still needs a layer
port and paired layer test. A-versus-B, C ownership, D's untested full ready queue and E's
PRQ win/XPIPE losses are unchanged. `layer_hp`, `layer_g51b` and `layer_m1rt51c` still
await their recorded tests; the layer release remains 2.014x. Do not add projection savings
to any family score. [Layer run log](../../../docs/runs-decode-layer.md), [release manifest](../../../docs/release-manifest.md).

Next evidence: preserve the existing A-versus-B stop rule and resolve the pending layer comparisons before claiming family or combined layer gains. For projection, retain paired controls and the ordering caveat when evaluating a layer port. No new experiments were launched by this review.

See the [tracker](../../../docs/architecture-performance-tracker.md) for prior evidence and combinations.

Validation: independent review checked the paired figures, percentages, receipts, attribution and ordering caveat against source files. Local links, dashboard filters, 24 comparison rows and four pending cards passed checks. Generated source fingerprint matches the tracker. The updated projection card was inspected in Chrome with no layout collisions observed. No benchmarks, submissions or pushes were performed.
