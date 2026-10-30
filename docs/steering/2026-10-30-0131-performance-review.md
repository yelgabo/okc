# Hourly review: standalone projection advances

Evidence snapshot: `87bc0b8`. Reviewed 2026-10-30 01:31 PDT.

The new projection round is separate context, not a gain attributed to A-E. The
[standalone roadmap](2026-10-30-attention-projection-standalone-roadmap.md)
separates standalone promotion from layer integration. A-E standings and the 128/256-token
A-versus-B decision remain unchanged. Direct cuBLASLt and router geometry changes have not
been ported to the layer. The full layer still needs its own controlled comparison.

The [projection run log](../runs.md#2026-10-30-standalone-round-direct-cublaslt-and-router-geometry-p1)
records these jobs. Routing distribution is not applicable to these standalone dense operators.
Scores below compare against each job's SGLang sample, not a same-process release control.

| Job | Source revision / candidate | Score (95% CI) | Evidence status |
|---|---|---|---|
| `bf6c825bd5cf4659` | `eb374e4`, LATE_ARRIVE baseline, submitted by coordinator | 1.457 (1.436-1.479) | Receipt verified, 9/9 |
| `1b00244922654709` | `d61deea`, P1a | 1.465 (1.443-1.487) | Receipt verified, 9/9; Lt build failed, router changes only |
| `047bc3a4168e42d6` | `33daf8b`, P1b, Lt heuristic search | 1.463 (1.442-1.485) | Receipt verified, 9/9 |
| `d6eaf1e73db047a5` | `62a6ae0`, P1c, tile/cluster sweep | 1.484 (1.461-1.506) | Receipt verified, 9/9; prepare-time picks vary |
| `44fa08c6d2f44b44` | `d0dc80f`, P1d, cross-layout search and longer screening | 1.479 (1.457-1.501) | Run-log evidence; receipt absent from checkout, correctness count unverified here |
| `6cbd8b981a5d4c3f` | `d0dc80f`, clean candidate, file SHA1 `03389e0d` | 1.482 (1.460-1.504) | Run-log evidence; receipt absent from checkout, correctness count unverified here |

Verified receipts: [baseline](../../muse_bf16_proj/profiles/proj_eb374e4.bf6c825bd5cf4659.json),
[P1a](../../muse_bf16_proj/profiles/bench_p1a.1b00244922654709.json),
[P1b](../../muse_bf16_proj/profiles/bench_p1b.047bc3a4168e42d6.json),
[P1c](../../muse_bf16_proj/profiles/bench_p1c.d6eaf1e73db047a5.json).

P1d's log reports 100 seeded rounds, medians against preparation A's pre-Lt pick.
Independent preparations newA/newB save 1.87/2.99 us on pooler 128 and 2.98/2.46 us on
pooler 256. Null deltas are -0.13/-0.03 us respectively. Qkv 256 saves 1.92/1.86 us;
unpooler 128 saves 1.34/1.41 us. Percent deltas are unavailable from this table because
absolute paired control times are omitted. These are component comparisons, not layer gains.
Pooler 128 picks vary by about 1.1 us between preparations; no paired confidence interval is
recorded in the table. Full per-shape deltas, including near-ties and router gaps, remain in
the linked run log.

The clean candidate's cross-run verify-16/verify-32 totals (128/256 tokens) are
73.29 -> 70.22 us and 91.84 -> 84.29 us: -3.07 us (-4.19%) and -7.55 us (-8.22%).
Decode totals vary from -0.45 to +0.62 us with unchanged decode code. These observations
cannot establish an isolated change or be added to A-E gains. Preparation rises from about
70 to 105 seconds. [Port status and cost](../proj-port-guide.md#direct-cublaslt-plans-lt_cpp-m--128--256-dense-2026-10-30).

Preserve the losses: Lt qkv 128/o 128 are near-ties in the searched space; several router
geometries lose outside their winning bucket; heuristic-only Lt search misses the pooler-256
winner. CUTLASS pooler 256 remains a tuner candidate but is superseded in this comparison.
P2 compressed qkv split-K is declined on a work model, without a new run; its reopening
condition is a one-wave geometry or persistent schedule that preserves startup and escape
cost. P3 compressed WGMMA is untested; its predicted gains are hypotheses. Neither is family
B or E evidence. [Scoped losses and reopening conditions](../ruled-out.md).

Next evidence: obtain the two missing receipts, stabilize the remaining pooler-128 pick
variation, and use the proposed raw-BF16 twin before judging P3's compression mechanism.
Standalone gains can qualify on their own; any layer port still needs a full-layer A/B.


The [tracker](../architecture-performance-tracker.md) retains the A-E status table and all
prior results. The dashboard uses the same evidence checkpoint. No kernel changes, GPU jobs,
submissions or pushes were performed by this review.
