# Post-reset steering: isolate the release, explain the remaining layer gap

Evidence snapshot: pushed `main` at **283e348c89e53b7c93b9928913dfab6293aabc68**.
The head commit is dated 2026-10-30 01:05 UTC, which is October 29 at 18:05 PDT.
Accordingly, the newest run-log rows say October 30 while this report uses the user's local date.

This is a source/receipt review and a proposed execution plan. No GPU runs, live quota query,
leaderboard query, kernel edits, or scored submissions were performed for this report.
Only this report and the steering index are published. Reconcile with newer commits before acting.

## Decision

**Retain the validated fused/hybrid architecture. Freeze the 1.982x release, separate the M1
experimental build from it, run a clean mode-47/projection comparison, and attribute the
remaining same-process verify-1/decode-32 gap before authorizing another architecture sweep.**

The previous integration and measurement round achieved its main goals. Do not repeat the
fat-wait repair, FIN8 coverage repair, initial public-layer comparison, or the staged-family
retest simply because older reports list them as pending. The new unresolved issue is M1's
replay validation failure, not a failure of the released 1.982x file.

## 1. Recorded state: release is not the current experimental source

| Item | Recorded evidence | Interpretation |
|---|---|---|
| Previous layer incumbent | `9b921febc7ad4140`, 1.964x, source `a923f11` | Historical control |
| New layer release | `edd71d3152a946fe`, **1.982383976x**, CI **[1.952648217, 2.012119736]**, 13/13 passed | Current scored incumbent in the committed records |
| Release confirmation | `261a76ac7b364091`, 1.989x; file SHA256 begins `1355d39f` | Exact file subsequently submitted as the new release |
| Same-process release versus previous incumbent | `aa4276e91c924dad`: 1.0158x speed ratio; candidate/null ratio about 1.0006 | Approximately 1.6% measured improvement in that benchmark |
| Same-process public versus release candidate | `d9e07c554f614d83`: public/candidate speed ratio 1.0464; null/candidate 0.9989 | Public reference still leads overall, but not on every load |
| Experimental follow-up | `e2d1d81447cc48d9`, 1.963x, 13/13 judge tests passed, internal `failed 1` | M1 replay stress failed, so M1 was disabled before timing; other follow-up changes still ran |

Sources: [layer run log](../runs-decode-layer.md),
[release receipt](../../muse_decode_layer/profiles/submission.edd71d3152a946fe.txt).

The release contains the route kernel, reworked single-token cgemv, MoE mode 37 and skinny
candidates including sk8w3s, qkv8, attention DCARRY/PK16/VHOIST, and the stress/diagnostic fixes.
The follow-up additionally contains mode 47, LATE_ARRIVE, o16/unp32 geometry changes, and the
SK_PDL/ONE_TRIG build for M1. At the evidence commit,
[`muse_decode_layer/submission.py`](../../muse_decode_layer/submission.py) still sets
`PORT_SK_PDL = True`.

**Do not equate HEAD with the scored incumbent.** The run log explicitly notes that the
SK_PDL build changes the projection's first-ring issue order even when the runtime M1 policy
is empty. Consequently, "M1 switched itself off" does not restore the release binary and does
not isolate the other changes. Nor should a future run be allowed to enable an unresolved
candidate merely because its one-time stress happens to pass.

Coordinator action: preserve the exact judged release file, its full SHA256, selected policies,
build switches, receipt and source provenance in a reproducible release manifest/snapshot.
Retrieve the full file hash from the saved artifact; the short hash above and the judge's payload
hash are not substitutes. Use the original non-M1 compile path as the control. Keep M1 on a
separate explicit experiment path until the failure is explained. This report does not itself
change any flags or delete experimental work.

## 2. What is now completed

### Integration and validation

The main bundle has been judged, confirmed and submitted. FIN8 stress now asserts mode 11 with
eight copies for both batch-1 full decode and verify; the committed successful run records
2,000 queued replays, zero replay-versus-eager difference for those cases, and clean workspace
invariants. Fresh-qkv checks and the speculative-length fallback were addressed as well.
The fallback investigation compared against a restated FP32 reference rather than merely
loosening the SGLang-distance bound. See `3e2b5771d7254fd2` and the pre-check audit section of
[the layer log](../runs-decode-layer.md).

The whole-layer benchmark now uses randomized arm ordering, a candidate duplicate/null arm,
held-out inputs, independent correctness checks and stage-health summaries. Failed early
attempts are recorded rather than treated as valid comparisons: `135ab2a9f22b49a5` had a
degraded incumbent due to stripped dispatch markers, and `121dff3ab17a4245` did not execute the
public bench. Their comparison numbers must stay excluded. The repaired comparison jobs are
`aa4276e91c924dad` and `d9e07c554f614d83`, each with 72/72 ABCHECK passes.

A 2.012x judge reading in one of the earlier bench jobs is not a reproducible new release score.
The scored receipt is 1.982x; use paired evidence, not the highest point estimate.

### Compiler/community leads

The recorded nvcc 12.8.1 attention audit did not reproduce the community's nvcc-13 trap/C7512
claim. Removing traps in a diagnostic build did not produce the claimed serialization cure.
Traps stay. This is a closed lead for the audited build, not a claim about every compiler version.
PDL provenance was documented for the changed route/MoE/attention paths; the report explicitly
says older unchanged PDL secondaries were not all re-audited. Do not advertise a universal
correctness proof. Source: [pre-check audits](../runs-decode-layer.md).

The fat-MoE first-try wait-accounting defect was repaired before this round. The resulting
mechanism study progressed to mode 47. Do not assign the same measurement repair again.
Source: [fat-wait steering](2026-10-29-fat-moe-waits.md) and [MoE run log](../runs-moe.md).

### Family B was actually retested

The 16:27 report correctly said B's revisit trigger had been met. That instruction is now
completed: `9d94ac75f4724b52` tested staged modes 45 then 46 on mode-37-era components.
The staged pair was **30.3 / 30.4 / 24.4 us slower** than mode 37 at **32 / 128 / 256 tokens**,
with none of the six seeds faster at each size. The same check found a small gain from mode 43,
then the controlled follow-up found mode 47 better without the hoist.

**Park this staged decomposition again.** Retain a revisit trigger, but require a mechanism that
changes the staged-versus-fused trade, such as a materially different phase kernel or dependency
cost. A small improvement that benefits both designs is not by itself evidence that the measured
24–30 us loss has disappeared. This does not reject all possible asymmetric/grouped designs.
Source: [MoE run log](../runs-moe.md), jobs `9d94ac75f4724b52` and `304f17f3e8704992`.

## 3. Replace the old cross-run priority map

Use the same-process whole-layer result `d9e07c554f614d83` as the current comparative map.
The table uses the run log's normalized weighted load times, `(46 * sliding + 16 * full) / 62`.
These are benchmark-arm times, not the official receipt's per-shape medians.

| Load | Our candidate, us | Public arm, us | Our excess, us | Our time above public | Share of net log gap |
|---|---:|---:|---:|---:|---:|
| Decode 1 | 86.9 | 88.8 | **-1.9** | -2.1% | -8.0% |
| Decode 16 | 157.1 | 151.1 | +6.0 | +4.0% | 14.4% |
| Decode 32 | 211.8 | 195.3 | **+16.5** | **+8.4%** | **29.9%** |
| Verify 1, 8 tokens | 126.6 | 116.7 | **+9.9** | **+8.5%** | **30.0%** |
| Verify 16, 128 tokens | 305.8 | 290.8 | +15.0 | +5.2% | 18.5% |
| Verify 32, 256 tokens | 400.6 | 384.5 | +16.1 | +4.2% | 15.1% |

Ratios and rounded shares are arithmetic over the logged arm times. The negative decode-1
share means our lead offsets losses elsewhere; it is not negative execution work.

**About 60% of the net same-process gap is verify 1 plus decode 32.** Decode 1 is already ahead
of this reference arm. Protect that gain rather than continuing to chase a historical decode-1
shortfall that the paired comparison does not show.

Important limits: the comparison used two held-out input sets per shape with eight rounds each,
including busiest/second-busiest local expert ranges; the null constrains noise in that run, not
all possible inputs. No confidence interval for the 4.64% comparative gap is provided here.
The pinned public blob is `936fefffb493c4a836e8c08fc58dff50ce338dd5`; its identity with the old
historical judged job `f662ee18` is not established. Use this as a direct implementation
comparison, not a live leaderboard prediction or proof of universal superiority.

## 4. First performance action: a clean follow-up without M1

### Mode 47: keep the useful mechanism

Mode 47 loads weight codes in four-step groups, reducing live code/scale registers from 18 to
10 and reported spill stores/loads from 156/416 B to 4/88 B. It does **not** retain mode 43's
next-stage hoist. Corrected wait measurements did not support claiming that the hoist reduced
waits. The measured success is the grouped-code/no-hoist implementation.

The pre-registered held-out gate, `5f381b8b6946492d`, measured mode 47 minus 37 on layer routing
at 32/128/256 tokens as **-1.0/-2.3/-3.7 us**; it also passed standalone and skewed routing gates,
FP32 correctness and spin checks. In the layer follow-up it passed mixed-size stresses and was
selected at 128/256 with front/PDL validation, but that run bundled several other changes.
Source: [MoE log](../runs-moe.md), [port guide](../moe-port-guide.md), [layer log](../runs-decode-layer.md).

### Projection changes: real, but small

The new o16 and unp32 geometries save about 0.3 us on their respective targeted projection calls.
LATE_ARRIVE at MT=2/UPW=2 adds about 0.2–0.5 us of per-call improvement on qkv32/o32/unp32 in
paired comparisons, retaining the fences. These are not explanations for a 16.5 us whole-load
gap. Preserve their matching packing/dispatch and test them without the SK_PDL build changes.
Source: [projection log](../runs.md), `2048b59099f3435d`, `f14c880c19a34369`,
`1093e22ca2534d7f`.

### Proposed bounded comparison

From the exact 1.982x release, prepare four named arms in the layer owner's worktree:

- R: exact release, including its non-M1 build configuration.
- R+47: only the mode-47 port; keep the old mode as a candidate/control.
- R+P: only the o16/unp32/LATE_ARRIVE projection package.
- R+47+P: the combined safe package, still without M1/ONE_TRIG/SK_PDL.

Use a null duplicate and random arm order, identical input sets, state isolation and visible
policies. First measure with controlled candidate choices at the relevant buckets to separate
kernel changes from autotuner flips; then validate the actual prepare-time policy. Do not force
mode 47 at 32 tokens merely because it wins at 128/256. A variant losing the test is dropped,
not silently assigned credit from another arm.

If file-size/compile/prepare limits make four arms impractical, split into two comparisons with
the same hashed R control. Do not reintroduce the marker-stripping problem. Require healthy
stage manifests and parity of generated code before interpreting timing.

Promotion: all correctness and replay gates pass; paired targeted improvements exceed the
null and repeat across held-out inputs; no unexplained material load regressions; the complete
layer improves. Do not require a tiny, well-attributed kernel change to produce disjoint
whole-score confidence intervals, and do not use a high isolated score as proof either.

Illustrative value, not a prediction: applying the mode-47 layer-routing gate deltas to the
new receipt gives about +0.36% layer speedup; a transfer of roughly 0.3 us at decode 16 and
1.1 us total at decode 32 for the projection package gives about +0.12%. Interactions and tuner
choices can change both. These are useful gains to test, not evidence of a large new architecture.

## 5. M1: unresolved validation failure, not a measured speedup or a diagnosed race

M1 is the attention-to-output-projection/front PDL overlap experiment, not a replacement MoE
scheduler. In `e2d1d81447cc48d9`, bit-16 launch self-tests passed, then the forced-on 1x8 full
replay stress exceeded its 1e-4 bound after 2,000 replays: hidden difference **2.0e-4**, stream
difference **1.6e-4**, equal routing IDs. The runtime policy became empty and chain tuning did
not execute. Therefore there is no measured M1 throughput benefit in that job.

The numbers alone do not establish stale reads or a data race; FP32 atomic order, rounding,
different compiled paths and numerical amplification are alternative hypotheses. Conversely,
passing the judge with M1 disabled is not validation of M1. Do not raise the tolerance or label
it harmless nondeterminism without identifying the first divergent intermediate and checking
the independent reference and workspace/reset behavior.

Give one owner a bounded diagnostic task, isolated from performance promotion:

1. Use the same compiled experimental binary to compare M1 off/on, each in eager and queued
   replay execution. Add off/off and on/on repeats to measure each path's own variability.
   Separately compare experimental-off with release R, since compilation itself changes order.
2. Fix input contents, selected plans and policy choices for the reproduction; isolate mutable
   cache/workspace state between arms. Assert the failing 1x8 full plan and exercise normal and
   supported fallback cases. Do not allow a differently selected mode to count as coverage.
3. Locate the first divergence: attention output/cache, o projection, boundary, route/pre-norm,
   MoE, final boundary. Record element counts, absolute/relative error, BF16 ULPs where meaningful,
   NaN/Inf and counters. Check each arm against the independent reference.
4. Do not retain only the final replay result. Use a bounded retained-output ring or device-side
   error summaries for sampled/intermediate replays without inserting a host sync after every
   replay. Exercise fresh captured-buffer contents as well as repeated inputs.
5. Revisit pre-wait data provenance, compiler clobbers, cache coherence, publication and reuse
   on the path implicated by the first divergence. Preserve fences and prove any ordering change.

Stopping rule: if one focused diagnostic run plus a targeted reproduction does not explain the
failure, leave M1 parked for the next performance round. If a real synchronization defect is
found, correctness repair takes precedence over that budget. If it is proven numerical rather
than ordering-related, document the evidence and approve any validation-policy change
explicitly before performance testing. No M1 promotion while attribution is unresolved.

## 6. Second main action: attribute the remaining whole-layer gap

The direct comparison has answered "is there still a gap?" It has not answered which component
or dependency owns all of verify-1's 9.9 us and decode-32's 16.5 us. The ABTL timelines were
lost from the log tail. Commit `f350ba4` adds compact ABTOP reporting and explicit null ratios,
but there is no later comparison result using that output in the evidence reviewed here.

Run a targeted version of the corrected whole-layer bench on verify 1 and decode 32, sliding
and full, preserving the public source pin and exact current release/control. Print compact
per-stage boundaries and the selected variants last. Keep full-step, non-profiled paired times
as the decision metric; profiler rows can include dependency waits and instrumentation effects.
Do not assign summed-row residuals to idle time.

Required output:

- End-to-end paired deltas and null for each of the four target shapes over additional held-out
  inputs; then score-weighted aggregation. Repeat local expert ranges/routing skew coverage.
- The actual T=32 MoE selection, since sk8w3s/main-lean/main-lean-silu-bar have flipped between
  runs. Compare fixed-policy arms to reveal a policy loss before changing the kernel.
- Arm routing agreement, local pair counts, distinct experts and job sizes. A legitimate
  numerical routing difference can change the work; distinguish this from execution speed.
- Non-overlapping critical-path segments around attention->o, front->route->MoE and MoE->unpooler,
  plus per-stage durations, not just the five longest kernel names.

If normal timeline attribution is insufficient, replay one captured stage input through both
compatible component implementations with identical routing/precision boundaries, or use one
correctness-checked diagnostic component swap inside a common layer schedule. Keep the full
production-path comparison authoritative and avoid mixing incompatible layouts or rounding
contracts. Public-source comparison artifacts remain measurement-only, not submission candidates.

The deliverable is a causal hypothesis with a measured recoverable budget. Only then pick the
next optimization: dispatch calibration, in-layer resource interaction, a named dependency
boundary, or a specific kernel's remaining work. Do not infer that PDL or a new GEMM algorithm
must own the gap simply because individual standalone kernels seem close to the public ones.

## 7. Architecture-family decisions

| Family / direction | Current decision |
|---|---|
| A: fused/hybrid MoE | Keep as incumbent. The release has mode 37; mode 47 is the next isolated candidate. |
| B: staged grouped phases | Retested on updated components and lost. Park; record the new receipt and a specific differential revisit trigger. |
| C: output ownership | Already part of the successful fused line. Not a new untested family to rebuild. |
| D: dynamic scheduling | Runtime phase-B claiming remains part of the incumbent. No evidence here justifies a full ready-job queue rewrite. |
| E: attention producer/consumer | DCARRY/PK16/VHOIST are integrated in the release. No new broad attention rewrite is needed for the present gap. |
| M1 PDL overlap | Separate correctness investigation; timing gate did not run. |
| Generic projection L2 side-branch prefetch | Remains parked under existing negative evidence. Do not conflate it with the narrowly scoped, already measured route-to-MoE PDL boundary. |
| Broad resource or geometry sweeps | Do not restart Humming-inspired or public-geometry searches without a measured new threshold; the logs now contain extensive negative coverage. |

These decisions supersede outstanding actions in the 16:27 report where the newer jobs
completed them. They preserve the older reports as historical evidence.

## 8. Execution order and resource discipline

1. Release owner: archive R and separate the experimental build; no GPU quota needed for that.
2. Layer owner: the clean R / R+47 / R+P / R+47+P comparison, then a confirmation of the selected
   package. Proposed budget: one or two paired checks plus one confirmation, subject to actual
   compile/size limits and live quota.
3. Measurement owner: one targeted public-layer comparison for verify 1/decode 32, with compact
   critical-path output and fixed-policy controls. Follow with at most one focused attribution run.
4. Correctness owner: M1 diagnostic in parallel on a separate file/worktree; never block the
   safe package on it. Coordinate GPU/remote allocations and avoid simultaneous timed kernels.
5. Coordinator: promote only the clean, validated winner under the existing user authorization;
   reserve confirmation/recovery capacity rather than exhausting layer checks on standalone sweeps.

One owner per problem directory; the layer owner integrates changes from component owners.
Query `okc problems` before dispatch because quotas are shared and the UTC reset has occurred;
old statements that every budget is exhausted are stale. I did not query live remaining quota,
so the counts above are proposed allocations, not available-run claims. Follow the repository's
actual `okc check <file> --no-wait --json` workflow and confirm returned job/problem IDs.

The next report should contain: exact release and candidate hashes; enabled/disabled feature
manifests; clean follow-up results; M1's diagnosed status; and the newly attributed verify-1 /
decode-32 time. Update tracker/dashboard status from evidence, not commit-message optimism.

## 9. Near-term goal and limitations

A reproducible 2x layer is close: from the scored 1.982384x point estimate it takes about +0.89%
speedup, or a uniform 0.88% time reduction. This arithmetic is not a claim that 2x is already
established; the scored CI spans 2x and run-to-run variation remains. The small clean ports alone
are not a guaranteed crossing. Do not search for a lucky score instead of a repeatable mechanism.

No current qualifying rank/threshold was verified here. Do not convert the public-arm ratio into
a predicted official score by multiplying unrelated benchmarks. The paired workload map is
more useful for the next engineering decision than stale rank targets.

This report reviewed pushed commits, the source switches, component/whole-layer run logs and the
release receipt. It did not inspect unpushed work, execute the GPU tests, certify the full memory
model, or freshly verify the external community sources. Remaining uncertainties are explicit:
M1's root cause, in-layer attribution of mode 47 and the latest projections, and the exact
critical-path ownership of the two largest same-process gaps.

## Source map

- [Layer run log and pre-check audits](../runs-decode-layer.md): `3e2b5771d7254fd2`,
  `aa4276e91c924dad`, `d9e07c554f614d83`, `261a76ac7b364091`, `e2d1d81447cc48d9`,
  `edd71d3152a946fe`.
- [Scored release receipt](../../muse_decode_layer/profiles/submission.edd71d3152a946fe.txt).
- [Layer source](../../muse_decode_layer/submission.py): `PORT_SK_PDL = True` at the evidence SHA.
- [MoE log](../runs-moe.md) and [port guide](../moe-port-guide.md): staged retest
  `9d94ac75f4724b52`, no-hoist comparison `304f17f3e8704992`, mode-47 gate
  `5f381b8b6946492d`, shippable check `b50351288b504e81`.
- [Projection log](../runs.md): `2048b59099f3435d`, `f14c880c19a34369`,
  `1093e22ca2534d7f`; o16/unp32 geometry and LATE_ARRIVE evidence.
- [Rejection register](../ruled-out.md), [coordinator working method](../../.claude/agents/okc-coordinator.md).
- [Previous 16:27 review](2026-10-29-1627-performance-review.md),
  [pre-bundle v1/d32 analysis](2026-10-29-v1-d32-breakdown.md),
  [fat-wait correction](2026-10-29-fat-moe-waits.md).

All source statements refer to the pinned evidence snapshot at the top, not an assumption that
relative links will remain unchanged on future main.
