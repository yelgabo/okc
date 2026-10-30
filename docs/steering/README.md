# Shared project steering

This is the stable entry point for steering reports across research, implementation,
and coordinator sessions. The user requested on 2026-10-29 that every project-steering
report be checked into this repository so other sessions can read it.

## Latest decision report: on course, with validation and prefetch priorities

**[2026-10-29: Course check — validate overlap and isolate prefetch](2026-10-29-course-check-validation-prefetch.md)**

Evidence snapshot `b41cb8d` (October 30 02:19 UTC / October 29 19:19 PDT), unchanged from
the preceding status answer. Keep the clean 2.014x release. Mode 51 passed its component gate
and awaits shippable/layer validation. The later 32-token MoE comparison measured only
1.5-1.9 us to the public kernel on layer-like routes; do not treat the earlier 10-14 us
profile attribution below as an established algorithmic deficit. Next: isolate the existing
MoE PDL prefetch hints, finish stage-level freshness/numerical diagnostics for M1, and keep
production validation thresholds unchanged pending an approved, evidence-based policy.
This report changes documentation only and launches no GPU work.

## Latest hourly report, 19:31

**[2026-10-29 19:31 PDT: Mode 47 layer attribution and mode 51 component gain](2026-10-29-1931-performance-review.md)**

Evidence snapshot `b41cb8d`. Mode 47 layer gains are isolated; mode 51 awaits layer testing.
Clean release scores 2.014x. M1's off control also fails the replay bound, so its earlier failure
is not isolated to M1. Tail-split modes 52/53 lose.

## Latest layer note: M1 PTX ordering argument (2026-10-30)

**[2026-10-30: M1, PTX ordering of the TMA reads around griddepcontrol.wait](2026-10-30-m1-ptx-ordering.md)**:
pre-wait reads have no live writer; x is read through TMA only after the wait, now with a
reader-side `fence.proxy.async.global` in every issuing thread. Results of the M1 timing and the
fresh-input gate are in `docs/runs-decode-layer.md` (6ce5be0ba29642ea, 60cf58b69b75491e,
fdf75ba5a8124544) and the M1 row of `docs/ruled-out.md`.

## Previous layer report: clean release, M1 diagnosis, verify 1 / decode 32 attribution

**[2026-10-30: Layer release, mode 47 isolation, M1 diagnosis and gap attribution](2026-10-30-layer-release-attribution.md)**

Board entry 3fbb3ad027e84c6b, 2.014x: the release plus mode 47, LATE_ARRIVE and o16 / unp32,
M1 off ([manifest](../release-manifest.md)). Mode 47 is worth 4-5 us at 128 tokens (720079090ee44cc3).
The M1-off layer itself breaks M1's 1e-4 replay bound, and no stale-read signature was found;
M1 timing awaits a validation-policy decision. The public layer's verify 1 / decode 32 lead sits
at the attention -> o and boundary -> front PDL weight-prologue boundaries (about 7-8.5 us) and
in MoE work at 32 tokens (about 10-14 us at d32 sliding).

## Previous decision report: post-reset release and next experiments

**[2026-10-29: Isolate the 1.982x release and explain the remaining layer gap](2026-10-29-post-reset-release-steering.md)**

Evidence snapshot `283e348` (October 30 01:05 UTC / October 29 18:05 PDT).
Release `edd71d3152a946fe` is 1.982x and distinct from experimental HEAD. Same-process public
comparison puts about 60% of the net gap in verify 1 and decode 32. The old integration,
FIN8 coverage, fat-wait repair and staged retest tasks are complete; B lost again. Next:
isolate mode 47 and projection updates from the SK_PDL build, diagnose M1's replay validation
failure without loosening its bound, and retain compact timing attribution for the two target
loads. The report makes recommendations; it did not run GPU jobs or change kernel code.

## Hourly report, 18:25

**[2026-10-29 18:25 PDT: Combined layer win and staged retest loss](2026-10-29-1825-performance-review.md)**

Evidence snapshot `283e348`. A/E + projection bundle wins by 1.58% in the paired layer
comparison; release scores 1.982x. B loses its retest; mode 47 wins component gates;
M1 overlap fails replay correctness before timing.

## Previous hourly report, 16:27

**[2026-10-29 16:27 PDT: Staged retest and assembled layer bundle](2026-10-29-1627-performance-review.md)**

Evidence snapshot `f61d8d9`. No new timing result. B's revisit trigger is met; combined
A/E and projection bundle `b694b4a` is assembled but unjudged. PDL overlap is compile-gated only.

## Previous hourly report, 14:25

**[2026-10-29 14:25 PDT: Attention component gains and validation gaps](2026-10-29-1425-performance-review.md)**

Evidence snapshot `efddc84`. PK16/VHOIST improves 128/256-token attention beyond DCARRY;
the combined package still awaits layer tests. Local commit under the automation's no-push rule.

## Latest community audit

**[2026-10-29: Verify 1 and decode 32 per-kernel breakdown](2026-10-29-v1-d32-breakdown.md)**

Evidence commit `efddc84`, scored layer run `9b921febc7ad4140` (1.964086x), pending bundle
`b694b4a`. After the bundle's measured standalone savings, no owned kernel trails the public
one by more than 1.6 us in the same process at verify 1; about 5 us (verify 1) and 7 us
(decode 32, inside the MoE row) weighted stay cross-run and unassigned. Ranks PDL weight
prefill at the attention -> o boundary first (+0.4% to +0.9%, hypothesis), a 32-token MoE
measurement second, the route-into-MoE fold third. Built `SK_PDL` (muse_bf16_proj) and
`ONE_TRIG` (attn_one.cu), both off by default with production SASS unchanged; no check run.

## Previous report

**[2026-10-29: Community research audit and updated steering](2026-10-29-community-audit.md)**

Evidence snapshot `21be06d`, with live quota, job receipts and board snapshots captured on
October 29. Integrate mode 37, sk8w3s, QKV8, the newer cgemv and attention DCARRY/PK16/VHOIST;
repair the whole-layer comparison and FIN8 selected-mode replay coverage. Fat wait accounting
is complete. Modes 43/44 remain a bounded proposal. All four relevant check budgets are
exhausted at capture; scored runs require explicit user approval.

## Previous hourly report, 13:27

**[2026-10-29 13:27 PDT: Projection context and unchanged families](2026-10-29-1327-performance-review.md)**

Evidence snapshot `615539c`. New single-token projection gains await layer integration;
A-E standings are unchanged. Local commit under the hourly automation's no-push rule.

## Previous hourly report

**[2026-10-29 11:20 PDT: Hourly performance review](2026-10-29-1120-performance-review.md)**

Evidence snapshot `14d6da1`, incorporating `ca43908f02874886`. Updates the tracker and dashboard
with corrected wait attribution. Family winners and layer-test gaps are unchanged. Local commit
only under the hourly automation's no-push constraint.

## Latest mechanism report

**[2026-10-29: Fat MoE wait accounting and the next mechanism](2026-10-29-fat-moe-waits.md)**

Evidence: code `3f67a45`, check `ca43908f02874886`. It closes section 2 of the overnight report
(the fat diagnostic's first-try_wait defect is fixed, production SASS unchanged), withdraws "fat
consumers never wait" (waits are about 20% of stage-loop time), and recommends one section 5
mechanism with a stopping rule. Nothing was built for it yet.
Addenda (2026-10-30): mode 43 built and dropped, 47 gated and kept; (3) warp skew measured
(warpgroups 750-1,200 cycles apart), phase gap to the public re-attributed, and mode 51
(final-pass sums of squares from 64 tokens up) passed its gate, check c6b0e500c3a84c9a.

## Previous report

**[2026-10-29 — After the overnight run](2026-10-29-overnight.md)**

Original evidence snapshot: `df03a59fc2ab02844e5003606e01b9a9de4bd5d2`.
Publication addendum checked against `d0c062b4fd5a5342cf6df678709e0af9b2632b9d`:
the three-slot skinny MoE experiment has since completed and produced candidate `sk8w3s`.
Read that addendum before assigning work from the original report.

The report recommends integrating validated component gains, repairing fat-MoE wait
accounting, and running a corrected matched whole-layer comparison. It records
recommendations, not execution of those tasks; check current code and run logs for status.

## Report history

| Report | Evidence snapshot | Scope |
|---|---|---|
| [2026-10-29 course check: validation and prefetch](2026-10-29-course-check-validation-prefetch.md) | `b41cb8d`; release `3fbb3ad027e84c6b` | Keep release; mode-51 integration; isolate MoE prefetch; stage-level M1 validation; reconcile smaller 32-token MoE gap |
| [2026-10-30 layer release and attribution](2026-10-30-layer-release-attribution.md) | branch `worktree-agent-a39812d9e0cf11e26` after `fa7200c`; scored `3fbb3ad027e84c6b` | Clean release manifest, mode 47 isolation, M1 fresh-input diagnosis, verify 1 / decode 32 exposed-time attribution against the public layer |
| [2026-10-29 post-reset release steering](2026-10-29-post-reset-release-steering.md) | `283e348`; release `edd71d3152a946fe` | Separate release/experimental HEAD, clean mode-47/projection test, M1 validation diagnosis, same-process verify-1/decode-32 attribution; staged retest completed |
| [2026-10-29 verify 1 / decode 32 breakdown](2026-10-29-v1-d32-breakdown.md) | `efddc84`; run `9b921febc7ad4140`; bundle `b694b4a` | Per-kernel breakdown, bundle credit, ranked mechanisms, `SK_PDL` / `ONE_TRIG` switches with compile gate |
| [2026-10-29 community research audit](2026-10-29-community-audit.md) | `21be06d`; live account capture 13:43 PDT | Research verification, completed work, pending ports, correctness gates, quota and next-run plan |
| [2026-10-29 13:27 performance review](2026-10-29-1327-performance-review.md) | `615539c` | Single-token projection context and unchanged family standings |
| [2026-10-29 11:20 performance review](2026-10-29-1120-performance-review.md) | `14d6da1` | Corrected fat-MoE wait attribution |
| [2026-10-29 overnight steering](2026-10-29-overnight.md) | `df03a59`; publication addendum `d0c062b` | Integration, diagnostic defect, whole-layer comparison, bounded research |
| [2026-10-29 fat MoE waits](2026-10-29-fat-moe-waits.md) | `3f67a45`, check `ca43908f02874886` | Repaired fat-MoE wait counters, overhead, corrected claims, next fat-MoE mechanism |

Earlier context is preserved in the [initial steering plan](../plan-2026-10-29-steering.md),
[architecture families](../architecture-families-2026-10-29.md), and
[historical gap map](../gap-map-2026-10-30.md). Their filenames are not proof that they are
newer than the evidence snapshot above. Original plans remain historical records.

## For the next session

Read the latest report and its evidence snapshot, then reconcile its recommendations with
newer commits and the component run logs. Do not repeat completed experiments just because
an older report lists them as next steps. Preserve the release incumbent and the user's
existing approval requirements for scored submissions.

## Publishing future steering

Every substantive steering/audit response must include a complete Markdown report under
`docs/steering/YYYY-MM-DD-<topic>.md`; add a time or sequence suffix for another report on
the same topic that day. Update this index's latest pointer and append the report to the
history without deleting earlier reports. Include the evidence commit, run identifiers,
measured results versus hypotheses, priorities, next actions, validation/stopping criteria,
and anything not verified. Mark subsequent corrections or progress as dated addenda or a
new report rather than silently rewriting the evidence history.

Commit and push the report and index in the same task, using the normal repository workflow
and preserving other sessions' work. Verify the remote files and report the path and commit
to the user. A local file or sandbox link alone is not a completed cross-session handoff.
If a push is blocked, say so explicitly; do not claim the other session can read it yet.
An explicit user request not to write to the repository takes precedence.

Hourly review history: [11:20 PDT performance review](2026-10-29-1120-performance-review.md).
