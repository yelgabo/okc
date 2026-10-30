# Shared project steering

This is the stable entry point for steering reports across research, implementation,
and coordinator sessions. The user requested on 2026-10-29 that every project-steering
report be checked into this repository so other sessions can read it.

## Latest hourly report, 16:27

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
