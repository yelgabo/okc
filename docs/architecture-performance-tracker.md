# Architecture performance tracker

Last review: 2026-10-29 22:34 PDT. Evidence snapshot: `6cb0796`.
First review reconciles progress since `aab7d75`, the snapshot in the
[architecture-family guide](architecture-families-2026-10-29.md).
This is a review of recorded experiments, not a new benchmark run.

## Current family status

Negative timing deltas mean faster. Percentages below are elapsed-time changes against the
named control, not changes in aggregate score. Families overlap; their gains cannot be added.

| Family | Current result | What has actually been measured | Visibility gap |
|---|---|---|---|
| A. Hybrid persistent/legacy MoE | Mode 51 now wins in the layer | Paired layer mode 51 versus 47 saves 5.1/5.3 us at 128 and 6.5/1.1 us at 256, sliding/full. | Final gate-version file has bench evidence but no unpatched scored confirmation or release; board stays mode 47. |
| B. Staged grouped MoE | Retested loss | Modes 45 + 46 on mode 37 components lose 30.4/24.4 us at 128/256 versus 37, six seeds each. | Parked until phase B alone drops below about 50 us at 128 or the fused phase A tail disappears. |
| C. Owner-completes tiles | Retained in the persistent descendants | Mode 20's owned phase A saves about 2-3 us at 128 and 4 us at 256 over mode 16. Later modes combine ownership with other changes. | Later gains cannot be credited to ownership alone. |
| D. Dependency-aware scheduling | Dynamic claiming retained; split-tail variants lose | Modes 52/53 split the last claimed phase B units through fixup and lose at 32/128/256. | Full ready-only scheduling remains untested; do not generalize the split-tail loss to the family. |
| E. Attention producer-consumer pipeline | Component gains; combined layer gain | DCARRY/PK16/VHOIST now runs in the winning layer package; corrected FIN8 plan/replay checks pass. | Package gain is shared with A and projection changes. M1 diagnostic timing gives little net release gain. Attention input-history dependence is an open, untested lead, not a confirmed E defect. |

## Latest review: mode 51 layer gain and measured M1 limits

Mode 51 closes its pending layer test. Job `29fd004fd58e40ef`, bench source `d9b7d67`, compares
`cand/layer_g51.py` (SHA256 prefix `dacd4e0d`, sent comment-stripped) with clean mode-47
release `b9c23bda`. Two held-out inputs per shape, seeds 140001+, eight cold-L2 rounds with
seeded random order give 16 paired samples per shape and 72/72 reference/routing/cache checks.
Candidate/release speed ratio is **1.0084**, null ratio 0.9997. At 128/256 only the MoE mode
differs in the timed path; gate changes affect prepare. Other shapes stay within null spread.

| Tokens | Sliding delta | Full delta | Control sliding/full | Null-minus-release sliding/full |
|---|---|---|---|---|
| 128 | -5.1 us (-1.92%) | -5.3 us (-1.52%) | 265.0 / 349.5 us | -3.7 / -5.2 us |
| 256 | -6.5 us (-1.86%) | -1.1 us (-0.21%) | 348.9 / 524.3 us | -4.4 / +0.3 us |

Percentages use paired delta divided by the logged control, not a subtraction of arm medians.
No per-delta confidence interval is reported. Modes at 32 are 47 in both arms, while fronts
differ at 16 through prepare-time selection. This supports the targeted layer gain, not a
new gain at every size. [Raw mode-51 layer comparison](../muse_decode_layer/profiles/benchG51.29fd004fd58e40ef.log).

The first gate-version candidate, SHA256 prefix `c8a843cf`, passes check `a438b82ff2c141c2`
at 2.022x (1.991-2.052), 13/13, with mode 51 selected at 32/128/256. The later `dacd4e0d`
file has the bench result above but no unpatched scored check. It is not released. Board entry
remains `3fbb3ad027e84c6b`, 2.014x. Mode 51 is bitwise at 128/256 in both arms of the
route-PDL checks. All 11 MoE variants pass; route gates pass 8/8; negative controls reject
tile, 1%, 0.1%, row and single-element injections in all 19 gates, and all uninjected
observations pass. [Release/candidate manifest](release-manifest.md), [layer runs](runs-decode-layer.md).

### M1: diagnostic gain within its build, little net release gain

Job `6ce5be0ba29642ea`, source `8f545ae`, runs M1 + mode 51 with a fresh-input gate and a
reader-side proxy fence. Judge 13/13 passes (1.988x, 1.959-2.018), but M1 is blocked on the
control at decode 1 sliding: off replay error 2.7e-3/2.5e-3, on bitwise. M1 changes no launch
at one token, so this does not isolate an M1 effect. Its timing gate did not run.

Stage diagnostic `60cf58b69b75491e` uses the same file through `dl_m1_stage_diag.py`, five
fresh-input pairs, 1,000 replays each for off/off2/on plus 200 eager calls. Attention relative
error is 1e-5 to 2e-4 in all arms; larger errors first appear at routing/MoE. No M1-specific
stale-read signature is found. Report-only timing now runs: at 8 tokens sliding, whole-step
off 105.9 versus o+front 100.5 us; at 16 sliding/full 153.4/229.1 versus 150.0/226.6; at
32, 217.7/360.1 versus 214.7/357.1. These are within-build diagnostic timings, not an accepted
candidate's release gain. Its judge score 1.965x (1.936-1.995) uses M1 off.

Measurement build `fdf75ba5a8124544`, generated with `dl_m1_bench_build.py`, compares M1 + 51
with the mode-47 release, seeds 130001+, eight rounds, 72/72 checks. It scores 2.007x
(1.977-2.038), paired aggregate 1.0040, null 0.9984. At <=32 tokens, candidate-minus-release
sliding/full is d1 +0.2/-0.3, d16 +0.6/+3.0, d32 -0.1/-0.2, verify 1 -1.3/+0.2 us.
Although turning M1 on inside this build saves 5.1 us at 8-token sliding, the build changes
plain projection issue order too, so net release gain is about zero at small sizes. Front
policies also differ at 16. At 128/256, mode 51 explains the targeted timing differences,
not M1. Gate is report-only: this build is not a validated candidate. No isolated E or D
family gain follows. [Layer run log](runs-decode-layer.md).

The same build's 32-token MoE prefetch probe finds off-minus-on +0.22/+0.00 us sliding and
-0.16/-0.48 full across two inputs, comparable to null -0.06/+0.26 and -0.32/-0.67. This
probe gives no measurable prefetch effect; it does not explain the earlier in-layer/component
gap. A prepared --no-pf-probe build and history diagnostic have no new recorded run.

### Validation changed; attention history remains an open lead

The [run log](runs-decode-layer.md#validation-policy-change-fresh-input-replay-gate-user-decision-2026-10-30)
records user-approved fresh-input gates for M1 and MoE/route replay checks, with later changes
to relative-L2 order-1 rejection, control envelopes, element checks and negative controls.
This review reports that recorded decision; it does not change validation or infer permission
for new work. The released file still uses the old gate; the candidate uses the newer one.
Attention `_one_stress`, norm and route side-stream bounds remain unchanged.

A [PTX ordering argument](steering/2026-10-30-m1-ptx-ordering.md) now exists and the M1 build
adds reader-side `fence.proxy.async.global`. The argument states that it has not had independent
review; runtime passing alone does not prove ordering. The former "argument missing" status
is superseded by "written, independently unreviewed".

New open lead in `5fefe98`: fresh-input attention differs from its eager reference by about
1e-4 at decode 1 sliding and 2e-4 at verify 1 sliding, while repeated-input stress is bitwise.
Previous-call dependence is a hypothesis, not a measured causal result or confirmed stale
read. `tools/dl_history_diag.py` is prepared to distinguish noise, previous-call dependence
and separate-case state; it has no run yet. This limits the earlier explanation that all
variation was atomic merge order. Do not clear M1 or reject E on this unresolved evidence.
[Open lead](runs-decode-layer.md#open-lead-2026-10-30-unproven-input-history-dependence-in-the-attention-output).

## Historical review, 19:31: mode 47 layer attribution and mode 51 component gain

Current layer release `3fbb3ad027e84c6b`, source `2cee446`, scores **2.014x (95% interval
1.984-2.045), 13/13 passed**. Its file SHA256 `b9c23bda16bb9d6311f025062e415841c67b16284737172badaa190fdc506a49`
matches main's layer file. It adds mode 47 plus LATE_ARRIVE/o16/unp32 projections to the prior
release, with M1 code absent. Check `c2ba945496ab47b4` was 2.006x (1.976-2.037).
These are cross-run scores, not an isolated gain estimate. [Release manifest](release-manifest.md).

### Mode 47's isolated layer contribution

Job `720079090ee44cc3`, bench source `fa7200c`, compares the scored file with the same file
holding mode 47 out of the picker (SHA256 prefix `84d8507a`). Both arms compile the same
kernels; prepare picks 47 versus 37 at 32/128/256. Two held-out inputs per shape, seeds
110001+, eight seeded-random cold-L2 rounds give 16 paired samples per shape; 72/72 checks
pass. Candidate-minus-held deltas, sliding/full: **128 tokens -4.9/-5.5 us (-1.81%/-1.43%);
256 -5.7/-3.6 us (-1.58%/-0.70%)**. Logged held controls are 270.2/383.7 and 361.2/514.8 us.
At 32 tokens -1.3/-2.6 us is inside the arm spread. This is one paired check, without a
per-delta confidence interval. Null-minus-held at 128 is -3.8/-4.0 and at 256 -4.8/-1.0 us.
[Raw attribution](../muse_decode_layer/profiles/benchInc_47.720079090ee44cc3.log).

The 1.0155 overall ratio is contaminated by a decode-1 arm-position artifact: the held arm is
about 5 us slower even though mode 47 never runs there. Do not report that aggregate as an
isolated mode-47 gain. Full-package comparison `50c4c1328ab742f4`, source `2cee446`, seeds
80001+, eight rounds, 72/72 checks, similarly prints 1.0182 but about 1.009 excluding decode 1.
Its paired deltas at 128 are -5.4/-6.1 us, at 256 -2.1/-3.8; projection share inferred by
subtracting across these two benches is only indicative. [Layer run log](runs-decode-layer.md).

### Mode 51 advances A; tail-split descendants lose

Mode 50 moves sum-of-squares work to a two-pass final pass. Job `8a7051fa926e43e1`, source
`a03ca00`, seeds 23-28, passes correctness but fails its 32-token layer-routing performance
gate (+0.7 us against a +0.5 limit), despite -5.9/-6.6 us at 128/256. It is not shipped.
Mode 51 uses that final pass only at 64+ tokens and retains mode 47 below. This is an A
implementation change retaining C ownership/D claiming, not a new isolated C or D result.
Gate `c6b0e500c3a84c9a`, source `69c06ed`, six held-out seeds 29-34 per routing:

| Routing | 32-token delta vs 47 | 128-token delta | 256-token delta | Seeds faster, 32/128/256 |
|---|---|---|---|---|
| Layer | +0.0 us | -3.9 us (-2.41%) | -3.9 us (-1.71%) | 2/6/5 of 6 |
| Standalone | -1.0 us | -3.8 us | -4.4 us | 5/6/6 of 6 |
| Skewed | -0.1 us | -4.0 us | -7.7 us | 2/6/6 of 6 |

Layer controls are 161.9/228.0 us. All 54 output comparisons pass (maximum 4.6e-5), self-test
3.88e-3, spin flags zero, judge 5/5. Shippable check `a5a2bbd1a18b4b89`, source `6bbb252`,
repeats -3.7/-3.0 us at 128/256, six of six faster, scoring 2.210x. Submission
`bcd2d7a3506a4b7d` passes at 2.202x, below mode 47's 2.216x board entry, which stays.
Those cross-run scores do not negate the paired gain. No mode-51 layer test is recorded.
[MoE runs](runs-moe.md), [raw gate](../muse_moe_w4a16/profiles/bench_g51.c6b0e500c3a84c9a.log).

Modes 52/53 split the last dynamically claimed phase B units into K halves via fixup, an
A/C/D descendant rather than a ready-only queue. Job `209cf29bf91c4ea2`, implementation
recorded in `260d791` (exact submitted source hash not pinned in the row), seeds 35-40:
52 minus 51 at 32/128/256, layer/standalone, is +3.3/+3.5, +5.9/+4.9, +8.5/+5.3 us;
53 is +8.1/+5.2, +7.3/+6.5, +9.2/+7.7. At 128/256 layer controls 159.2/216.0 give
52 +3.71%/+3.94%, 53 +4.59%/+4.26%. Output differences are at most 9.7e-5, spin flags zero,
zero to two of six faster. Tail does not shrink (7.9 versus 7.1 us at 32). Both lose; no
layer port or further gate. This does not reject every split-K design.
[Raw tail test](../muse_moe_w4a16/profiles/bench_tail52.209cf29bf91c4ea2.log).

### Corrections and diagnostic limits

M1 remains blocked and untimed, but the failure is not isolated to M1. Diagnostic
`6372154fcd7a4fa9` on source `6377b5a` finds repeated-input M1-on/off bitwise equal at verify
1 full; the M1-off control itself varies up to 6.7e-4 at 32x1 full. Fresh-input diagnostic
`3e2aa83358dd4f60` finds 217-459/2000 off replays and 288-307/2000 on replays above 1e-4 at
verify 1 full, maximum 1.2e-3 in both arms; no replay exceeds 1e-2 or changes ids. The
source report attributes this to layer run-to-run variation, consistent with atomic reduction
order. The first divergent intermediate and proxy-ordering argument remain unverified.
The 1e-4 bound is unchanged; no timing or safety conclusion follows. Validation-policy changes
require explicit user approval under the recorded test plan; this review changes none.
[Layer diagnosis](steering/2026-10-30-layer-release-attribution.md),
[raw fresh-input test](../muse_decode_layer/profiles/m1fresh.3e2aa83358dd4f60.log).

Public comparisons `86ced49f092a48d8` and `f5ee8632e1f64bf2`, current release against pinned
blob `936fefff`, each pass 72/72 with 16 paired samples per shape; raw null deltas are within 1.1 and 1.8 us,
respectively.
Public advantages at verify 1 sliding/full are 8.7/10.5 then 10.2/10.8 us; decode 32
15.7/10.8 then 12.2/15.9 us. Profiled boundary overlap suggests a mechanism, not measured
recoverable time. Component test `1fbb9fbf524f4ed7`, source `b3509a0` plus tool changes,
layer/standalone seeds 1-6, measures only 1.5/5.9 us public advantage at 32 tokens (47 controls
90.9/132.9 us); the follow-up tail test finds 1.9 us on layer routing. This does not reproduce
the 10-14 us in-layer profile gap. Routing, prefetch and profiling effects remain hypotheses.
sk8w3s loses 19.2 us versus 47 at 32 on layer routing, zero of six faster; do not generalize
its 1/8/16-token win. ABPUB stage columns are invalid without --pubranges; use ABWORK.
[MoE attribution](runs-moe.md), [layer attribution](steering/2026-10-30-layer-release-attribution.md).

Diagnostic `ed3b4fdc76b54263` (source `64cec1f`) crashed after 128-token timing; no 256 result.
Rerun `d47c78bdc130424b` (source `54318bd`) completes, public minus 47 -14.7/-22.6 us at
128/256; diagnostic mode 49 adds 5.8/9.4 us. Its mislabeled ABSKEW columns must be read in
skew_cols order. The estimated 72-74% wait contribution from warpgroup skew is attribution,
not recoverable layer savings. [MoE run log](runs-moe.md).

## Historical review, 18:25: combined layer gain, staged loss and mode 47

The A/E plus projection package now has same-process whole-layer evidence. Check
`aa4276e91c924dad` (bench source `6c6038b`, candidate SHA256 prefix `4ed09046`, incumbent
`a923f11`) reports incumbent/candidate load-time ratio **1.0158**: 1.58% speedup, about
1.56% less elapsed time. It tests 12 shapes, two held-out inputs each (seeds 50001+), eight
cold-L2 graph rounds with seeded random arm order. All 72 reference/routing/cache checks
pass. The identical-candidate null gives about 1.0006 relative to the candidate; per-load
spread is about +/-0.5%, not a formal confidence interval. [Raw layer comparison](../muse_decode_layer/profiles/benchInc.aa4276e91c924dad.log).

At 128 tokens, paired candidate-minus-incumbent is -6.2 us sliding (-2.20%) and -13.3 us
full (-3.56%); at 256 it is -3.5 us (-0.98%) and -22.5 us (-4.22%). Percentages use logged
control means 281.2/373.1 and 356.6/533.2 us. Other loads: decode 1 +0.2/-1.3 us,
decode 16 -1.9/-2.1, decode 32 +0.9/-1.0, verify 1 (8 tokens) -0.7/-0.7, sliding/full.
Those smaller effects are similar to the null spread. The package combines A mode 37/sk8w3s
(retaining C ownership and D dynamic claiming), E DCARRY/PK16/VHOIST, and qkv8/cgemv changes.
It does not isolate a family gain or justify adding component savings.

Release `edd71d3152a946fe` uses the confirmation file from `261a76ac7b364091` (SHA256 prefix
`1355d39f`): **1.982x, 95% interval 1.953-2.012, 13/13 passed**. Confirmation was 1.989x
(1.959-2.019); six candidate checks ranged 1.960-2.012, averaging 1.982. These cross-run scores
support repeatability but the paired comparison supplies the gain claim. FIN8 mode 11 with
eight copies now passes the asserted 2,000-replay tests, replay/eager difference zero in
`3e2b5771d7254fd2`. [Layer run log](runs-decode-layer.md).

### A versus B and the next A candidate

B's retest `9d94ac75f4724b52`, source `54257b5`, uses modes 45 + 46 (variant 301), mode 37's
components split into owned phase A then claimed phase B/final pass. Layer-routing seeds 1-6
lose +30.3/+30.4/+24.4 us at 32/128/256, zero of six faster at each size. At 128/256 this is
+18.4%/+10.7% versus logged mode 37 means 165.5/227.1 us. All arms pass output checks,
spin flags zero; no delta confidence interval is reported. B remains parked under the new
trigger in the status table. [Raw retest](../muse_moe_w4a16/profiles/bench_gw1.9d94ac75f4724b52.log).

Mode 43's next-stage hoist improves timing but does not reduce measured wait cycles. Repeat
`304f17f3e8704992` (source `a4687c8`, layer seeds 7-12) finds mode 47, code groups without the
hoist, faster at 128/256. This scopes the failed hypothesis to the one-group hoist; mode 47
retains the code-group change. Gate `5f381b8b6946492d`, source `20c295e`, seeds 13-18:

| Routing | 32-token delta | 128-token delta | 256-token delta | Seeds faster, 32/128/256 |
|---|---|---|---|---|
| Layer | -1.0 us | -2.3 us (-1.32%) | -3.7 us (-1.65%) | 6/5/6 of 6 |
| Standalone | -2.4 us | -2.8 us | -3.4 us | 6/6/6 of 6 |
| Skewed | -2.0 us | -2.2 us | -2.8 us | 5/6/6 of 6 |

All deltas are paired mode 47 minus 37; layer controls are 173.6/223.8 us at 128/256.
Outputs differ by at most 4.4e-5, self-test 3.89e-3, spin flags zero; 5/5 judge tests pass.
Shippable check `b50351288b504e81`, source `ec565a7`, repeats gains (-2.5/-3.4 us means at
128/256, seeds 17-22). Component release `abe7651dbaa44458` scores 2.216x. Mode 47 also runs
at 128/256 in layer check `e2d1d81447cc48d9`, but that file changes several mechanisms and
scores 1.963x (1.933-1.992); it does not establish an isolated layer gain for 47.
[MoE runs](runs-moe.md), [raw gate](../muse_moe_w4a16/profiles/bench_g47.5f381b8b6946492d.log).

### Invalidated comparisons and separate layer context

- `2cd4e3b292904662` (source `d006531`) passes 13/13 but trips the attention setup test and
  falls back to SGLang, scoring 1.785x. This is not the intended bundle's performance.
  Restoring original qkv rows for the fallback comparison fixes the test path in `3e2b`.
  Later FP32 diagnostics show the fresh-input discrepancy was larger in SGLang's path.
- `135ab2a9f22b49a5` has an invalid incumbent arm: comment stripping removed dispatch markers.
  Discard its apparent 23-240 us gains. `121dff3ab17a4245` never runs its public bench because
  Triton cannot retrieve the arm source. The repaired comparisons are `aa4276` and `d9e07`.
  Oversized/static-rejected attempts produced no job and no measurement.
- Public comparison `d9e07c554f614d83`, same bench source, seeds 60001+, passes 72/72.
  Public/candidate speed ratio is 1.0464, null/candidate 0.9989. Weighted public advantages
  are 9.9 us at verify 1 and 16.5 us at decode 32; candidate leads decode 1 by 1.9 us.
  This replaces the older cross-run gap inference. The public blob is pinned in the run log;
  its identity with the earlier public score's submitted file is not established.
- M1 (SK_PDL + ONE_TRIG), source `6377b5a`, job `e2d1d81447cc48d9`, fails forced-on replay
  correctness at verify 1 full: after 2,000 replays hidden 2.0e-4 and stream 1.6e-4 exceed
  the 1e-4 bound; ids match. M1 is disabled before timing, so no M1 timing gain/loss exists.
  This is an attention-to-projection chain failure, not rejection of E or D. Reopening needs
  an ordering explanation and clean replay checks. [Layer results](runs-decode-layer.md),
  [rejection register](ruled-out.md).

Projection context outside A-E: jobs `01703396ea1a409e`, `3678bfd3ef1c4ce6`,
`2048b59099f3435d` test o16/unp32 geometries (bench sources `dd67954`, `977afb3`, then
resulting geometry change `4d53658`), about 0.3 us gains across
six passes, cold L2, 200 randomized rounds per pass, ABCHECK passes and null within 0.10 us.
`c6e13cc974a24294` repeats against old geometries/public. `d2c9cb6997794859` rejects NACC2
(+0.00 to +0.78 us); `807464e240ec4f6c` finds fence/vote probes within noise and late release
losing +0.26 us on pool32. Fences stay. `f14c880c19a34369` and `1093e22ca2534d7f`, source
`eb374e4`, test LATE_ARRIVE: at 32 tokens qkv saves 0.38-0.48 us, o 0.22-0.26, unpooler
0.16-0.32 over nolate, across two passes each; module-null spread is -0.06 to +0.16.
The mixed M1-off layer check includes these changes but does not isolate their layer gains.
Detailed quartiles, control timings and negative geometry results remain in the
[projection run log](runs.md) and its linked profiles.

## Historical review, 16:27: staged retest and assembled layer bundle

No new scored run or paired timing appears in `dbd99c3..f61d8d9`. The
[rejection register](ruled-out.md) now parks the staged design with an explicit revisit trigger:
cheaper phase B/setup or a changed overlap tail. Modes 23, 26 and 28/37 landed after the old
staged comparison. That trigger is met, so repeat A versus B at 128/256 with current components
in one process. Preserve the old +27.9/+24.2 us losses as historical measurements; they do not
establish the result against today's A. No new winning B implementation has been measured.

The [layer breakdown](steering/2026-10-29-v1-d32-breakdown.md) reports an assembled bundle at
`b694b4a`: mode 37 + sk8w3s (A, retaining C/D mechanisms), DCARRY/PK16/VHOIST (E), plus qkv8
and single-token cgemv projection changes outside A-E. It is unjudged. Its branch log and
source were inspected locally; FIN8 expected-plan assertions and fresh-qkv replay checks are
implemented there, but no completed runtime result closes that coverage gap. Whole-layer
score remains 1.964x from `9b921febc7ad4140`. Do not add standalone savings into a layer gain.

Separate layer context: `3da50e5` adds default-off SK_PDL projection-weight prefill and
ONE_TRIG attention launch-dependent signaling. This is a proposed attention-to-projection
chain, not D's ready-job scheduler or a measured E gain. Recorded compile checks show the
disabled paths SASS-identical (58/58 projection kernels, 15/15 attention kernels); enabled
builds have no recorded spills. The host launch path was not compiled in that gate, and
runtime correctness, replay stress and timing remain untested. No job ID or timing delta
exists for this proposal. [Compile evidence and test plan](steering/2026-10-29-v1-d32-breakdown.md#6-built-sk_pdl-and-one_trig-compile-gated-off-by-default).

The new score-weighted hotspot tool in `f61d8d9` ranks existing profile evidence. Its rankings
are analysis, not fresh benchmark measurements. The layer breakdown's bundle projections
and remaining-gap estimates are also hypotheses or cross-run arithmetic, not isolated family gains.

## Historical review, 14:25: family E improves again

PK16 changes probability packing and VHOIST carries transpose addresses/descriptors per
segment. The combined attention candidate improves full verification beyond DCARRY.
Job `025ccaf9c57a4ba2`, switches introduced in `a50fc0c`, records cold same-process control
116.26 / 204.22 us versus candidate 110.91 / 194.59 at 128/256 tokens. That is **-5.35 / -9.63
us**, or **-4.6% / -4.7%**. Warm deltas are -7.52 / -13.88 us. ABCHECKSUM is 19 passed and
zero failed for every arm. These full-attention cases use batch 16/32, q_len 8, context
80,000 and TP8; MoE routing does not apply. The run-log table supplies no per-delta
confidence interval or independent repeat count. [Raw bench](../muse_attention_decode/profiles/bench_pk.025ccaf9c57a4ba2.log).

Candidate-only run `569b233dfe08414f`, source `c37d381`, passes 16/16 at 2.449x,
95% interval 2.413-2.502. Full verify times 103.65 / 187.67 us are 7.1 / 10.7 us below the
mean of the two previous DCARRY checks. Those are cross-run observations. Repeat
`e9085c86d89449c2`, source `61e3fd4`, uses the same candidate as its judged path and passes
16/16 at 2.504x, interval 2.467-2.542, with 100.83 / 182.24 us. Its machine is faster on
unchanged rows too; the extra headline-score increase is not another code improvement.
[Attention report](runs-attention.md).

Family E's pending package is now **DCARRY + PK16 + VHOIST**. No new layer result exists in
the reviewed records; layer score stays 1.964x. A-D standings and the mode 37/sk8w3s layer
gaps remain unchanged. The no-fence timing probe in the repeat is diagnostic only and is not
an eligible winning candidate; the production fence stays.

The [community audit](steering/2026-10-29-community-audit.md) identifies two validation gaps:
the intended q_len8 FIN8 stress selected mode 15 instead of mode 11 with eight copies, and
the whole-layer comparison lacks enforced per-seed acceptance, reliable arm-order balancing
and independent mutable cache state. These are open coverage/measurement gaps, not recorded
correctness failures of the incumbent. They should be resolved before relying on the next
combined layer comparison. Existing 13/13 layer judge passes remain recorded.

Decode-32 full attention's apparent +8.7 us regression in the older DCARRY comparison is
now attributed to SGLang's bimodal baseline statistic and machine variation on unchanged mode
49. Preserve the observed cross-run statistic but do not label it a demonstrated candidate
regression. [Rejection register](ruled-out.md), [attention analysis](runs-attention.md).

## Historical review, 13:27: projection context

No new A-E or whole-layer result appeared since the 12:24 review. The three family candidates
awaiting layer tests remain mode 37, sk8w3s and ONE_DCARRY. New projection results belong
outside A-E; they do not change the winning-family badges.

Job `f6c80b4ba11048f3`, source `a35709c`, compares the new single-token cgemv with the
pre-change file `b79a627` in one process. Two passes use different inputs, each with 300
randomly ordered rounds and cold L2. New minus control, pass 0 / pass 1:

| Single-token projection | Paired change |
|---|---:|
| qkv_gate | -0.54 / -0.45 us |
| pooler | -0.86 / -0.96 us |
| unpooler slice | -0.35 / -0.22 us |
| o | +0.08 / -0.11 us, inconclusive |

ABCHECK passes on each reported shape/pass. The same-code null stays within 0.13 us.
The implementation combines row-major planes, a single-load escape start and a larger
register allowance for two chunks per thread. It is a combined projection change, not a
MoE ownership or scheduling experiment. Absolute paired control means and intervals are
not quoted here, so no elapsed-time percentage is assigned.

The candidate-only submission `81254e8aed794eab` records **1.444x, 95% interval 1.422-1.466**.
This is the projection problem's score, not the layer's. Layer integration remains pending;
current layer score stays 1.964x. The earlier qkv8 projection port is also still pending in
the port instructions. [Projection report](runs.md),
[paired raw log](../muse_bf16_proj/profiles/bench.f6c80b4ba11048f3.log),
[candidate result](../muse_bf16_proj/profiles/submission.81254e8aed794eab.json).

The proposed 128-token wide concat front was declined without a new run. Its estimated
24.4 us best-case time exceeds cuBLAS's recorded 24.0 us, before accounting for a second CTA
wave. This is an analytical rejection of that geometry, not a measured candidate loss and
not evidence against staged MoE B. [Rejection register](ruled-out.md).

## Historical review, 11:20: corrected MoE wait measurements

Job `ca43908f02874886`, code `3f67a45`, corrects a diagnostic defect. The timer previously
started after the first potentially blocking try-wait, which could hide waiting entirely.
The earlier claim that fat-path consumers never wait is invalidated. Production instructions
are unchanged, so this is corrected attribution, not a new production speedup or regression.

| Diagnostic observation | 128 tokens | 256 tokens |
|---|---:|---:|
| Phase A weights + activation waits, per consumer warp | 15.73 us | 23.26 us |
| Phase A waits / stage-loop time | 19.6% | 20.2% |
| Phase B waits / stage-loop time | 20.9% | 20.8% |
| Instrumented mode 41 minus production mode 37 | +7.0 us, +4.3% | +6.1 us, +2.7% |

Six paired layer-routing seeds; production controls 164.6 / 227.2 us. The run passes 5/5
judge tests. Counter overhead is material, and per-warp waits cannot be summed into layer
savings. The producers still spend about 70% of phase A waiting for free ring slots.
[Raw log](../muse_moe_w4a16/profiles/bench_fw1.ca43908f02874886.log),
[measurement report](runs-moe.md#fat-path-waits-with-the-first-try-timed-2026-10-29).

A/C/dynamic-claiming production wins remain supported. B remains a tested loss; no new C,
ready-only D, E or whole-layer candidate result appeared in the reviewed changes. The same
three candidates await layer evidence. A proposed per-group load/hoist experiment remains
unbuilt in the [steering report](steering/2026-10-29-fat-moe-waits.md); its predicted savings
are hypotheses. This correction motivates a test, not a new winning-family badge.

## Large-token comparison ledger

| Family / candidate | Control | 128 tokens | 256 tokens | Evidence and scope |
|---|---|---:|---:|---|
| B: staged 19 then 21 | Mode 16 | +27.9 us, +11.2% | +24.2 us, +8.4% | `e221858b020c4dee`, six same-process seeds, complete standalone MoE. Controls 249.4 / 287.3 us. Correctness passed. [Staged gate](runs-moe.md#staged-gemm1-gate-2026-10-29). |
| C inside A: mode 20 | Mode 16 | -2.1 us, -0.8% | -3.8 us, -1.3% | Same job and controls. Other paired checks support the sign. This isolates owned phase A within a fused launch. [MoE log](runs-moe.md). |
| D-related inside A+C: mode 23 | Mode 20 | -16.2 us, -6.6% | -16.1 us, -5.7% | `a9dbec1d98334943`, source `10375cb`, six paired normal seeds; controls 246.2 / 283.7 us. Self-tests pass, spin flags zero. [Dynamic phase B](runs-moe.md#phase-b-claimed-at-run-time-mode-23-2026-10-29). |
| Mode 23 held-out repeat | Mode 20 | -22.9 us, -8.7% | -18.7 us, -6.1% | `36812632dae242ba`, source `ec219f0`, six held-out normal seeds; controls 263.3 / 304.6 us. Skewed routing also improves, -4.8 / -22.9 us. [MoE log](runs-moe.md). |
| A+C+dynamic claiming: mode 26 routing/gather changes | Mode 23 | -5.8 us | -9.5 us | `817eab6401d54a84`, source `ba9a457`, six held-out seeds; self-tests pass. Skewed -4.2 / -7.7 us. [MoE log](runs-moe.md). |
| Same combined design: mode 28 consumer/ring and sorting changes | Mode 26 | -10.6 / -11.6 us | -11.6 / -12.5 us | Two six-seed checks `f928f8a802014e4f` and `13fe68e7ca2e423f`, source `ec62b5a`; correctness passes. [Consumer and ring investigation](runs-moe.md#consumer-stage-and-the-ring-slot-releases-2026-10-29). |
| Same combined design: mode 37 phase A epilogue | Mode 28 | -3.2 / -4.3 us | -4.9 / -5.3 us | `987d024620e44eb7` and `6307d0c91c8d4ada`, source `96f2c43`, held-out seeds 17-22. Correctness passes. Reciprocal and publication-order changes, not a new ownership experiment. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29). |
| E: ONE_DCARRY, cold paired bench | Same attention without DCARRY | -3.65 us, -3.0% | -7.33 us, -3.5% | `4ff339332fef4aff`, full verify only, controls 120.03 / 210.59 us; all arms pass checksum checks and judge 16/16. [Attention bench](runs-attention.md). |
| E: ONE_DCARRY, judged repeats | ONE_PFAST pair without DCARRY | -3.6 us, -3.2% | -7.9 us, -3.8% | Cross-run mean versus mean, `f64e8680b3e441be` / `bf49eceb15a64f81`, source `0d76acc` / `503c0f4`; 16/16 both. Component evidence, not layer savings. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, cold paired bench | DCARRY control | -5.35 us, -4.6% | -9.63 us, -4.7% | `025ccaf9c57a4ba2`; controls 116.26 / 204.22 us, candidate 110.91 / 194.59. ABCHECKSUM 19/0 for every arm. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, candidate-only judged | Mean of previous DCARRY pair | -7.1 us, -6.4% | -10.7 us, -5.4% | `569b233dfe08414f`, source `c37d381`; 16/16 passed, cross-run component observation. Same-code repeat `e9085c86d89449c2` also passes but has machine-speed variation. [Attention log](runs-attention.md). |
| B: staged 45 + 46 retest | Mode 37 | +30.4 us, +18.4% | +24.4 us, +10.7% | `9d94ac75f4724b52`, six layer-routing seeds, outputs pass; zero faster. [MoE log](runs-moe.md). |
| A+C+D: mode 47 code groups | Mode 37 | -2.3 us, -1.32% | -3.7 us, -1.65% | `5f381b8b6946492d`, held-out layer seeds 13-18, 5/6 and 6/6 faster; checks pass. [MoE log](runs-moe.md). |
| A/E + projection: whole-layer bundle | Incumbent a923f11 | Sliding -6.2 us (-2.20%); full -13.3 us (-3.56%) | Sliding -3.5 us (-0.98%); full -22.5 us (-4.22%) | `aa4276e91c924dad`, 2 inputs per shape, 8 rounds, 72 checks pass; combined gain only. [Layer log](runs-decode-layer.md). |
| A: mode 47 isolated layer picker | Same file with 47 held out | Sliding -4.9 us (-1.81%); full -5.5 us (-1.43%) | Sliding -5.7 us (-1.58%); full -3.6 us (-0.70%) | `720079090ee44cc3`, 16 paired samples/shape, 72 checks pass; overall ratio invalidated by d1 arm artifact. [Layer log](runs-decode-layer.md). |
| A: mode 51 final pass | Mode 47 | -3.9 us, -2.41% | -3.9 us, -1.71% | `c6b0e500c3a84c9a`, layer seeds 29-34, 6/6 and 5/6 faster; no layer test. [MoE log](runs-moe.md). |
| A/C/D: mode 52 split tail | Mode 51 | +5.9 us, +3.71% | +8.5 us, +3.94% | `209cf29bf91c4ea2`, layer seeds 35-40, output checks pass. Mode 53 also loses +7.3/+9.2 us. [MoE log](runs-moe.md). |
| A: mode 51 isolated layer | Clean mode-47 release | Sliding -5.1 us (-1.92%); full -5.3 us (-1.52%) | Sliding -6.5 us (-1.86%); full -1.1 us (-0.21%) | `29fd004fd58e40ef`, 16 paired samples/shape, 72 checks pass, 1.0084 aggregate with null 0.9997. Candidate not released. [Layer log](runs-decode-layer.md). |

Percentages use the logged paired delta divided by the logged control mean. Paired summaries use logged paired deltas rather than subtraction of separately summarized
arm timings. Layer deltas are medians of per-round differences; these can differ by several
microseconds from differences of arm medians. Rows without a comparable control mean report
microseconds only. Seed counts are reported instead of inventing confidence intervals for the
paired differences.

## Small-token results and rejected changes

For A's new `sk8w3s`, job `75bca9775f6f4b0b`, source `03887be`, uses six seeds per size and
routing distribution. This measures the MoE component using layer-like or standalone routing;
"layer-like" does not mean a whole-layer benchmark.

| Tokens | Layer-like control sk8w4s | Change with sk8w3s | Standalone-routing change |
|---|---:|---:|---:|
| 1 | 35.6 us | -0.9 us, -2.5% | -0.6 us |
| 8 | 66.2 us | -2.4 us, -3.6% | -1.8 us |
| 16 | 84.8 us | -3.1 us, -3.7% | -2.9 us |

All 36 paired seeds favor three slots. Every arm matches the control within 2.1e-5 and spin
flags are zero. The candidate entered the standalone variant list at `d0c062b`.
[Evidence and port status](runs-moe.md#skinny-weight-stream-against-the-public-kernel-2026-10-29).

Losses matter as much as wins:

- A's unhinted skinny weight copies lose 1.2-3.3 us across the six size/routing rows in that
  job. Early refill with three slots also loses in every row. Scale-decode and helper-handoff
  changes in `e937013292b24906` were inconclusive and not adopted.
- Phase A mode 35 reduces spills but loses +3.0 / +4.7 us at 128/256 versus mode 28.
  Mode 39's paired stores lose +1.0 / +0.2 us versus mode 37 in `473d87d03de94e96`.
  A shorter epilogue alone does not establish a faster kernel. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29).
- E's run-time pair weighting loses +9.0 / +12.0 us at full verify 256 for weights 1.40 / 1.50
  in `4ff339332fef4aff`. The arrival counter fails to predict scheduler priority. This is an
  attention scheduling loss, not evidence against MoE family D. [Attention log](runs-attention.md).

## Whole-layer outcome and attribution

Board release remains `3fbb3ad027e84c6b`, mode 47, 2.014x (1.984-2.045), 13/13. Mode 51
has targeted paired layer gains in `29fd004fd58e40ef`; its exact final gate-version file still
needs unpatched scored confirmation and is not released. M1 is now measured diagnostically,
with little net release benefit at <=32; report-only validation prevents treating it as an
accepted candidate. B/C/D standings are unchanged. [Manifest](release-manifest.md).

## Decision and next evidence needed

Keep the clean release while confirming the final mode-51 candidate. Resolve attention
input-history dependence with the prepared controlled diagnostic before assigning all variation
to atomics or changing further validation. Independently review the written M1 ordering
argument. Do not count its within-build speedup as a release gain. B remains parked after
its retest; modes 52/53 remain losses and D's full ready-only queue remains untested.
[Rejection register](ruled-out.md).

## Review history and checkpoint

- 2026-10-29 22:34 PDT: reviewed `22264ae..6cb0796`, five new layer jobs
  `6ce5be0ba29642ea`, `60cf58b69b75491e`, `fdf75ba5a8124544`, `a438b82ff2c141c2`,
  `29fd004fd58e40ef`, manifest, validation changes and unrun history diagnostic. Other
  component checkpoints unchanged. Existing 18 untracked JSON profiles are already logged
  jobs. Origin/main fetched with no remote-only commits. Board release unchanged; mode 51
  now layer-tested, final-file confirmation pending.

- 2026-10-29 21:29 PDT: checkout remains `22264ae`; fetched origin/main with no
  remote-only commits. Four run logs, rejection register and steering index are unchanged.
  All 18 untracked profile JSONs are already logged jobs. No new measurement or correction;
  mode 51 still awaits layer testing. Review metadata refreshed locally, no commit or push.

- 2026-10-29 20:31 PDT: inspected through `22264ae`; fetched origin/main with no
  remote-only commits. Four run logs and rejection register are unchanged since `b41cb8d`.
  All 18 untracked profile JSONs are already logged jobs. Accessible chat inventory supplies
  no newer relevant evidence. Run checkpoints, family standings and mode 51's pending layer
  test are unchanged. Review metadata refreshed locally; no new-evidence commit.

- 2026-10-29 19:31 PDT: reviewed `b7da8bb..b41cb8d`, release manifest, four run logs,
  rejection updates and raw layer/MoE diagnostics. Latest layer release `3fbb3ad027e84c6b`,
  attribution `720079090ee44cc3`; latest MoE submission `bcd2d7a3506a4b7d` (board stays 47),
  tail test `209cf29bf91c4ea2`. Projection/attention standalone checkpoints unchanged.
  All new run IDs and invalidated diagnostics are listed above. Existing 18 untracked profile
  JSONs already correspond to logged jobs. Origin/main fetched, no remote-only commits.

- 2026-10-29 18:25 PDT: reviewed `b75e2f9..283e348`, four run logs, rejection updates and
  raw MoE/layer comparisons. Latest layer `edd71d3152a946fe`, MoE `abe7651dbaa44458`,
  projection `1093e22ca2534d7f`; attention component remains `e9085c86d89449c2`.
  All new jobs are listed above, including invalidated attempts. All 18 untracked profile
  JSONs are already logged jobs and were preserved. Fetched origin/main, no remote-only
  commits. Dates on source logs use 10-30; this review is 10-29 in America/Vancouver.

- 2026-10-29 17:22 PDT: inspected through `b75e2f9`; fetched origin/main with no
  remote-only commits. All four run logs and the rejection register are unchanged since
  `f61d8d9`. All 18 untracked profile JSONs correspond to already logged jobs. No new
  measured result or layer confirmation; run checkpoints and A-E standings are unchanged.
  Refreshed review metadata locally; no commit because no new evidence was found.

- 2026-10-29 16:27 PDT: inspected `dbd99c3..f61d8d9`, all four run logs, the updated
  rejection register, compile-gate reports and bundle `b694b4a`. Fetched origin/main with
  no remote-only commits. All 18 untracked profile JSONs are already logged jobs.
  No new timing jobs; newest attention remains `e9085c86d89449c2`, layer remains
  `9b921febc7ad4140`. B retest trigger is met; bundle assembled but unjudged.

- 2026-10-29 15:24 PDT: inspected through `dbd99c3`; fetched origin/main with no
  remote-only commits. Run logs and rejection register are unchanged since `efddc84`.
  All 18 untracked profile JSONs correspond to already logged jobs; no new provisional
  measurement found. Accessible chat inventory shows no newer relevant activity.
  A-E standings, pending layer tests and run checkpoints are unchanged. Refreshed review
  metadata and generated output locally; no commit because there is no new evidence.

- 2026-10-29 14:25 PDT: reviewed `615539c..efddc84`, including attention jobs
  `025ccaf9c57a4ba2`, `569b233dfe08414f`, `e9085c86d89449c2` and community-audit validation
  findings. Fetched origin/main; no remote-only commits. New E gains, no new layer result.
  Newest incorporated attention run is `e9085c86d89449c2`; other stream checkpoints remain.
  Existing untracked profiles preserved. Local commit only under the no-push rule.

- 2026-10-29 13:27 PDT: reviewed through `615539c`; fetched origin/main, no remote-only commits.
  Incorporated projection runs `5ff76a62b0044378`, `f6c80b4ba11048f3` and
  `81254e8aed794eab`, sources `803178f` / `a35709c`, reported in `754549d`, plus
  `f88e278`'s analytical concat rejection. A-E logs and layer outcomes unchanged. Untracked
  profiles remain untouched. Updated dashboard context without assigning projection gains
  to a family. Commit is local under this automation's no-push rule.

- 2026-10-29 12:24 PDT: no new experiment evidence since the 11:20 review. Fetched origin/main;
  no remote-only commits. Run logs and tracked profiles unchanged; all 18 untracked result
  JSONs are readable and their run IDs already appear in existing reports. Family outcomes,
  pending layer tests and latest incorporated run IDs remain unchanged. Refreshed review
  metadata only; no new evidence commit under the automation's commit policy.

- 2026-10-29 11:20 PDT: reviewed `d0c062b..14d6da1`, fetched origin/main and found no remote-only
  commits. Incorporated diagnostic run `ca43908f02874886`, source `3f67a45`, reported by
  `eced63d`. No new layer result. Prior candidate timing ledger retained. Dashboard summaries
  and review metadata updated; no kernel, benchmark or submission action. Commits remain local
  because this automation explicitly forbids pushing.

- 2026-10-29 10:24 PDT: initial reconciliation through `d0c062b`. Read current MoE, attention
  and layer run records, architecture baseline and layer build selection. Recorded wins and
  losses above without running GPU work or modifying implementation files.
- Latest incorporated streams: MoE skinny `75bca9775f6f4b0b`; MoE fat `6307d0c91c8d4ada`;
  attention `4c12b0199efd4b8c`; layer `9b921febc7ad4140`. These streams have independent
  histories; use run IDs and source revisions rather than treating this list as time order.
- Untracked profile JSONs were inventoried and readable. Their corresponding historical runs
  already appear in committed logs; they were not staged. This review uses the committed
  experiment reports, not an independent recomputation of every raw sample or memory-ordering
  proof. Later reviews should inspect changed reports and new run IDs, including uncommitted
  results, and explicitly flag any disagreement.
- Superseded "pending" and "not submitted" paragraphs remain in historical logs. Later named
  result rows control the current status. The old MoE footer still names mode 23's board entry;
  the later mode 37 submission row supersedes it. No live leaderboard was queried.

## Visual dashboard

Open the [local dashboard](http://127.0.0.1:4387/session/3de14ef757fbea5b) while Lavish is running,
or open the [standalone HTML](../.lavish/architecture-performance/index.html). It uses the
existing gap-analysis palette and typography stack. Filters separate measured wins, tested
losses, partial designs and candidates awaiting layer evidence.

Hourly reviews update this tracker and `docs/architecture-dashboard.json`, then run
`python3 tools/build_architecture_dashboard.py`. The builder derives the comparison table
from this tracker and embeds its fingerprint. Keep JSON summaries, pending candidates and
review timestamp aligned with the evidence. Layout changes belong in
`tools/architecture-dashboard-template.html`; rebuild and check them in the browser.
