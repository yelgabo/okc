# OKC steering brief after the overnight run

Evidence snapshot: pushed `main` at `df03a59fc2ab02844e5003606e01b9a9de4bd5d2`.
This is a steering recommendation, not a report of new GPU work.
Latest recorded scored layer in that snapshot: `9b921febc7ad4140`, source `a923f11`,
1.96408566x, 95% CI [1.93462438, 1.99354695], 13/13 tests passed.

## Publication addendum — 2026-10-29

This report preserves the preceding research-session recommendation and its evidence snapshot;
it is not a fresh audit of every commit made afterward. The shared entry point for this and
future reports is [the steering index](README.md).

At publication, the repository had advanced to
[`d0c062b`](https://github.com/yelgabo/okc/commit/d0c062b4fd5a5342cf6df678709e0af9b2632b9d).
That commit adds the three-slot skinny MoE candidate `sk8w3s` after run `75bca9775f6f4b0b`:
36 of 36 paired seeds were faster than `sk8w4s` across 1/8/16 tokens and layer/standalone
routing. The recorded layer-routing deltas are approximately -0.9/-2.4/-3.1 us, respectively.
The refill-order and no-L2-hint variants did not win. See
[the MoE run log](../runs-moe.md#skinny-weight-stream-against-the-public-kernel).

Therefore the skinny ring experiment proposed in section 5 below has already been performed.
Check current integration status and validate `sk8w3s` as a layer candidate rather than
repeating that experiment. This is component evidence, not a claimed new layer score.
The rest of the report remains tied to `df03a59`; reconcile recommendations with newer run
logs before acting. No GPU benchmark or kernel change was made while publishing this report.

## Decision

Keep the current fused/hybrid architecture. The next round is integration, measurement
repair, and a matched whole-layer comparison—not another unconstrained mode sweep.
The gap-map file at the evidence snapshot is a historical snapshot of 1.930x and several of
its proposed experiments have already run.

## 1. Preserve the incumbent and integrate finished component work

Preserve the exact source, generated device-code hashes, compiler flags, selected policies,
and results of the 1.964x layer. Do not equate the repository head with the scored source.

Three component changes are not yet in that scored layer:
- MoE mode 37, `main-lean-silu-bar`, from standalone source `96f2c43`.
  Paired standalone evidence supports roughly 3–4 us at 128 tokens and 5 us at 256;
  layer-routing comparisons also support a small improvement at 32.
  The layer inliner at the evidence snapshot lists `(0, 16, 26, 28)`, not 37.
  Read `docs/review-mode37-ordering-2026-10-29.md`.
  Preserve writer and issuer `fence.proxy.async.global`, the CTA barrier, the
  release on `done_as`, and consumer acquire. The removed generic GPU fence is not
  the proxy fence. The review's SiLU comparison used newer SGLang source; verify the
  exact pinned baseline and required BF16 boundaries before asserting exact parity.
- Attention `ONE_DCARRY=1`, from `0d76acc` / comment-only repeat `503c0f4`.
  Checks `f64e8680b3e441be` and `bf49eceb15a64f81`: full verify 16/32 improve
  about 3.6/7.9 us. `ONE_WAVE` remains off. Do not count the earlier PFAST
  decode-32 cross-run change as a new demonstrated DCARRY benefit.
- QKV at eight tokens, `SK_GEOM["qkv8"]` from `fee7249`:
  four units per warp per stage, eight stages, four-slot ring.
  Paired improvement ~0.66–0.69 us; submitted run `826cab6b81aa4f81`.
  Regenerate matching geometry and packing; do not reuse an incompatible packed stream.

Use the actual layer front/PDL environment for validation. The layer MoE now has
a pre-wait barrier initialization/prefetch patch not present in the standalone.
Check all new Params fields, resource use, barrier arrival counts, and generated-source
parity. Keep each existing fallback until the new candidate wins.

Validate each change's attribution, then the combined layer. Include repeated graph replays,
fresh inputs, mixed-size consecutive calls, scratch invariants, and explicit assertions
that tests select the intended variants. In particular, FIN8 coverage must actually select
q_len=8 mode 11 with eight workspace copies, not a shorter case routed to mode 15.
Do not repeat a test already demonstrated in current logs; inspect its selected mode.

An illustrative transfer model puts these three ports near +0.8% layer score, about 1.98x
from 1.964x, not a measured result and not a guarantee of 2x. It excludes the later skinny
candidate described in the publication addendum.

## 2. Repair a remaining diagnostic accounting defect before new fat-MoE conclusions

At the evidence snapshot, `muse_moe_w4a16/submission.py`, around lines 733–757:

    if (mbar_try(bar, parity)) {
        return 0u;
    }
    const long long t0 = clock64();

This is `mbar_wait_cyc`, used by `mbar_wait_m` for phase-A diagnostics
(modes 31, 33, 38, 40). `mbar_try` issues `mbarrier.try_wait.parity`.
That instruction can suspend even on the first attempt. A successful first try
therefore does NOT establish a zero-duration wait; a failed first try also
has elapsed time that this function discards.

The skinny diagnostic was fixed, but the fat diagnostic still uses this pattern in the
audited source. The old statement that fat consumers never wait for weights/x is not
established by these counters. Measured end-to-end gains remain valid.

Time the complete wait, including the first try, or sample complete waits.
Measure instrumentation overhead against the exact uninstrumented variant;
check SASS, registers and spills. Keep instrumentation out of the production path.
Audit other try_wait counters for the same mistake. Do not blindly replace try_wait
with test_wait: that changes the measured execution behavior.

Primary external reference: NVIDIA PTX ISA 9.4,
`mbarrier.test_wait / mbarrier.try_wait`: try_wait is potentially blocking.

## 3. Run the whole-layer paired comparison now

Use `tools/dl_make_bench_pub.py` as a starting point. Do not defer this until after
another component-optimization round.

Compare the frozen incumbent, integrated candidate, and pinned public layer on
identical inputs and weights, with independent scratch/cache state and cold-L2 graph replays.
Use multiple independent held-out seeds, randomized or counterbalanced arm order,
an ours-versus-ours null arm, and all six loads when limits permit. Report routing
agreement, local route/expert counts, and correctness before timings.

Repair the harness before relying on it:
- `ABCHECK` currently prints differences rather than enforcing acceptance; make it fail
  on real invalid output and validate more than seed zero.
- Do not use literal scored-case seeds as the only evidence. Hold out independent seeds
  and routing distributions; retain only allowed shape/config policies.
- The two-arm order expression (rotate, then reverse on odd rounds) produces the same
  order repeatedly. Use explicit AB/BA or seeded random permutations.
- Keep data mutated by one arm from contaminating another arm's input.
- Handle `ABSTOP` explicitly; a skipped shape is missing coverage, not a passing comparison.
- Respect prepare and run limits; split diagnostic batches if necessary.

Compute load time as `46 * sliding + 16 * full`, then the geometric mean of six
baseline/candidate load ratios. Use whole-step timing as the decision metric.
Timelines locate exposed dependencies; summed profiler rows are not wall time,
and judged-minus-profile is not an idle-time budget.

This experiment should answer where decode-32 actually loses and whether the
small-token whole-layer gap survives a same-process comparison.

## 4. Updated prioritization, provisional

Against the old public profile, the 1.964x layer's remaining log-gap shares are:
verify 1 (8 tokens) ~28%, decode 32 ~24%, verify 32 ~17%, verify 16 ~16%,
decode 16 ~11%, decode 1 ~4%.

These are arithmetic on a single current run and a historical public reference.
They identify a shortlist, not component-level causal proof. Replace them with the
matched whole-layer result.

At the evidence snapshot, the public skinny-MoE comparison was measured:
T=1 ours is at least competitive; T=8 public leads ~3.5 us; T=16 ~2.2 us
on the layer-routing bench. Closing that particular gap translates to
roughly 0.7% of layer score under direct transfer, not the prior speculative 1–2%.
The later three-slot-ring results in the publication addendum supersede those gaps for
that candidate; they do not yet establish its whole-layer impact.

## 5. Research after the corrected measurement

Authorize one bounded mechanism per lane, not a new family merely because it sounds promising.

Fat MoE:
- First determine exposed waits versus computation with repaired counters.
- If consumer live-register pressure is the limiting mechanism, test loading/dequantizing
  smaller code groups per stage instead of holding all stage codes live.
  Check spills, instruction dependencies, and complete-MoE time. The losing volatile-shared
  reload experiment (mode 35) is not the same mechanism, but must not be repeated.
- A natural-k intermediate layout with coalesced 16-byte writes is a separate, higher-risk
  option because it changes the phase-B operand layout. It needs a traffic estimate and
  data-layout proof before implementation.

Skinny MoE (original recommendation; see the publication addendum before acting):
- The ALU scale-decode and partial-buffer handoff variants were null; do not port them.
- Dynamic useful instruction counts are already near parity with the public kernel.
- Corrected waits point to the interaction of refill scheduling, ring depth and compute.
  A bounded ring/refill experiment is legitimate; stop if it cannot recover a useful
  part of the measured 3.5 us at eight tokens on held-out inputs.
- Early release is only a small layer-distribution candidate and loses standalone;
  do not promote it universally.

Layer/attention:
- The route-fusion experiment, one-token GEMV rewrite, and DCARRY work have already run.
  Do not repeat old gap-map tasks.
- Prioritize a component only if whole-layer savings can justify its experiment budget.
- Keep tested staged decomposition, unchanged pair-balancing proposals, blanket PDL,
  unfenced candidates, and rejected arithmetic/barrier variants closed.

## 6. Budget and project target

Check actual remaining quotas first. Logs report exhaustion of the layer's check
and scored budgets; do not assume new capacity until confirmed.
At the next available allocation reserve explicit integration, whole-layer diagnostic,
and confirmation capacity before launching exploratory sweeps.
Do not consume the last layer run while validated component ports are still waiting.

Track "validated standalone", "integrated", "layer-validated", and "submitted" separately.
Keep a single release manifest and append-only evidence log.
Missing self-test/tuning lines in truncated logs require an explicit compact summary.

The historical 2.115x target was a rank-100 snapshot in the gap map.
Official serving priority is top 50 on the overall kernel leaderboard.
Refresh the actual qualification threshold through the official CLI; the public
web leaderboard was unavailable during the original audit.
Use 2x as an engineering milestone, not a qualification claim.
From 1.964x, 2.115x would require ~7.7% more speedup / ~7.1% uniform latency reduction.

## Evidence files and runs

All original evidence below is pinned to `df03a59fc2ab02844e5003606e01b9a9de4bd5d2`;
the publication addendum separately cites `d0c062b`.

- `docs/runs-decode-layer.md`: latest `9b921febc7ad4140` in that snapshot and integration history.
- `muse_decode_layer/profiles/submission.9b921febc7ad4140.txt`: exact times and CI.
- `docs/gap-map-2026-10-30.md`: historical 1.930x map, now partly superseded.
- `docs/runs-moe.md`: mode37, small-size public comparison, corrected skinny stamps.
- `docs/review-mode37-ordering-2026-10-29.md`: independent ordering review.
- `muse_moe_w4a16/submission.py`: remaining fat diagnostic first-try omission.
- `tools/dl_moe_inline.py`: modes and layer-only PDL patch at the snapshot.
- `docs/runs-attention.md`, commit `6abd1dc`: DCARRY results.
- `docs/runs.md`, commit `32d8d96`: qkv8 geometry results.
- `tools/dl_make_bench_pub.py`: whole-layer comparison harness requiring the checks above.
