# Stable prepare-time picks in the decode layer (design, 2026-10-30; not implemented)

Diagram: [plan-2026-10-30-tuner-stability.html](plan-2026-10-30-tuner-stability.html).

## Problem (measured)

prepare() picks a policy per token count from timings it takes on the judge GPU. Two copies of
the same file in one process pick differently. In the four-arm benches 724aa0dd6ed140ca (A/A),
2dc566298be1498f (mode 51) and 40b9b3f795f84073 (M1 runtime), ABPOL showed DIFF between the
two copies of one file in these tuners:

| tuner | where it flipped (same file, two copies) | how it decides today |
|---|---|---|
| `_front_tune` | 8 tokens skf against skf-r (all three runs); 16 tokens skf against skf-r (2dc5) | one 30-replay median per variant (route variants: front + MoE chain), sequential, no margin |
| `_abnd_tune` | 16 tokens c8x256 against sglang (40b9, 2dc5); 1 token c8x256 against c1x1024 (2dc5); 256 tokens c1x512 against sglang (40b9) | six rotated rounds, median of round medians, 0.3 us margin for the incumbent c8x256 only |
| `_moe1_tune` | 32 tokens mode 47 against mode 37 (40b9, inc against inc2) | mean over 4 routings of one 20-replay median each, variants timed back to back, no margin |
| `_w4_tune` | GEMM configs at 1 / 8 / 16 / 256 tokens (all three runs) | summed medians over 2 routings, no margin |

The W4 flips have no timing effect while every token count picks a fused MoE variant (all
three runs): the W4 chain is then a fallback only. The MoE flip has a measured cost: in
40b9b3f795f84073 inc2 picked mode 37 at 32 tokens and its d32 sliding ran +3.1 us against inc
(null inc2 / inc 1.0128 on the d32 load). The 3fbb3ad027e84c6b receipt timed 47 at 99.7 us
against 37 at 101.5 us at 32 tokens, so both picks sit within about 2 us.

Two effects follow. A scored run inherits whichever pick its prepare() lands on, so a candidate
can lose or gain about 1% on one load by chance. In the four-arm bench the flips widen the nulls,
which are the yardstick of the decision rule, so real gains of 0.5-1% read "not resolved".

## What the picks are for

The goal is the fastest policy on the judge's GPU. The tuners exist because the judge GPU,
driver and clocks could differ from what we measured offline. A flip between two options
whose true times differ by less than the tuner's noise costs at most that difference. What
matters is (a) never picking an option that is truly slower by more than the noise, and
(b) picking the same option every run when two are tied, so that runs are comparable.

## Alternatives

1. **More rounds and a decision margin with a fixed preference order (the requested design).**
   Every tuner times its candidates in R interleaved rounds with the order rotated each round
   (as `_abnd_tune` does), takes per round the paired difference of each candidate against the
   preferred one, and switches from the preferred candidate only when the median paired
   difference saves more than a margin and at least R - 1 of the R rounds agree on the sign.
   The preference order per token count is the receipt policy of the current board entry
   (3fbb3ad027e84c6b: MoE sk8w3s / 47; front fp32 / skf-r / concat-r; abnd as its manifest row).
   The margin is max(0.3 us, 2 x the median absolute deviation of the preferred candidate's
   round medians), so it scales with the measured noise of that shape.
2. **Pinned policy with a safety check (eliminates the noise).** Hard-code the receipt policy
   and keep the tuners only as a guard: each pinned pick is timed against the alternatives
   once and replaced only when it loses by a large margin (for example 3 us or 3%), which only a
   different GPU or a broken kernel produces. Zero flips on the judge's GPU by construction, and
   a few seconds of prepare saved. Costs: a new component's pick must be set by hand from
   bench evidence, and a pick that is truly better by 1-2 us on the judge is never found
   at run time.
3. **Keep the tuners, pool the bench.** Leave prepare() alone and give the bench `--dup=shared`
   for the policy (one prepare's policy forced onto both copies). This cleans the nulls but
   leaves the scored run's own variance. Measurement-only.

Recommendation: 1 for the tuners whose options are close (front, abnd, MoE at 32 tokens),
with the preference order from the receipt, so that a tied comparison always lands on the same
option. Alternative 2 is the stronger structural change and is worth testing first if quota is
short: it removes the variance entirely at no prepare cost, and the receipt policy is what
scored 2.014x.

## Expected variance reduction (estimate, not measured)

Model: a pick between two options with true difference d and per-decision noise s flips with
probability p about Phi(-d / s). The load time variance it adds is p (1 - p) d^2.

- Today, d is under s for the flipping options above (that is why they flip), so p is near
  0.5 and the added standard deviation is about d / 2 per load: about 1.5 us at d32 (MoE 47
  against 37, d about 2-3 us, load about 220 us: 0.7% of that load) and 0.5-1 us at the front
  and abnd shapes (0.3-0.8% of d1 / v1 loads, 0.2-0.4% of d16).
- Across the six loads, which the score averages in the log domain, the flips contribute a
  score standard deviation of roughly 0.15-0.3%. This is small against the 1-2% run-to-run
  swing of scored runs, which comes mostly from the per-run MoE routing and the bimodal SGLang
  baseline. It is comparable to the four-arm nulls (0.35-1.03%), so the bench benefits more
  than the board.
- With design 1 (R = 9, margin 2 MAD), a tied comparison lands on the preferred option unless
  8 of 9 rounds favour the other by more than the margin: p drops from about 0.5 to under 0.05
  for a true tie, and the added variance by about 10x. A real 2 us win still switches when
  2 MAD is under 2 us. The tuners' round-to-round spread was never logged; the cand2 - cand
  nulls of the four-arm benches, mostly 0.0-0.8 us per whole step outside d32f and v32f,
  suggest it is under 1 us, but it has to be measured per tuner.
- With design 2, the added variance is zero on the judge's GPU.

The A/A arm effect at d1 (inc slower by 4-6 us in all three four-arm runs, with an identical
policy at 1 token in 724aa0dd6ed140ca) is not a policy flip. Stable picks will not remove it;
it is the module-identity question of docs/steering/2026-10-30-layer-bench-arm-bias.md.

## Cost

Prepare time: today `_front_tune` takes one 30-replay median per variant; R = 9 rounds of 20
replays is about 6x the front's replays (it times 2-4 variants at 6 token counts: an estimated
+5-10 s). `_moe1_tune` at 4 routings x R rounds: an estimated +10-20 s. The 40b9b3f795f84073
bench reached ABEND at 166 s with four prepares, and the single-file check a438b82ff2c141c2
spent 146 s in prepare, so an extra 30 s fits under the 300 s limit. This has to be checked with ABEND
and the judge's prepare time, not assumed.

## Cheapest decisive test

One check with a bench file in which both copies are the same file built with the stabilized
tuners (an A/A of the new file): ABPOL should read "same" on every line except W4, and the
nulls should be at most about 0.4% outside d1. Compare against 724aa0dd6ed140ca (old A/A:
three tuners DIFF, null 0.35%). Stop if ABPOL still shows DIFF in front, abnd or MoE, or if
prepare grows by more than 40 s.

## Gaps

- The per-round spread of each tuner was never logged; the margins above are guesses from the
  bench's ABTIME medians. The first implementation should print the per-round values
  (TUNE lines are cut from the 16 KB tail, so a compact SUMMARY line per tuner).
- Whether the judge GPU differs between scored runs is unknown. If it does, a pinned policy
  (design 2) can be wrong on some runs; design 1 adapts.
