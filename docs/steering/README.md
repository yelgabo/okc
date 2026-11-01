# Shared project steering

This is the stable entry point for steering reports across research, implementation,
and coordinator sessions. The user requested on 2026-10-29 that every project-steering
report be checked into this repository so other sessions can read it.

## Latest checkpoint: top-10 audit and Windows restart

**[2026-10-31: Top-10 audit and restart handoff](2026-10-31-top10-restart-checkpoint.md)**

Evidence `8f2c89b`; live layer rank371, best2.018705x against observed rank10 at2.379322x.
Top10 needs17.864% more score; the existing experiment portfolio does not establish that gain.
Official snapshots, current per-shape hotspot/score reconstruction, reconciled backlog and
restart steps are saved. WSL2.7.14 installed; Windows requires restart before Linux/okc setup
can finish. No kernel changes or new GPU jobs. Receipt-analysis tools fixed and4/4 tests pass.

## Family F registration, historical policy

Current policy supersedes the adoption language below. All four tracks now have measurements,
and donor code is measurement-only. Reimplement winning mechanisms in owned code.
[Policy addendum](2026-10-30-family-f-donor-baseline.md#addendum-2026-10-31-user-decision-on-the-use-of-f-results).

**[2026-10-30: Family F — Donor-based baseline and hybrid](2026-10-30-family-f-donor-baseline.md)**

Registered at the user's request against evidence snapshot `7a59655`; donor
`alygg77/okc-muse-spark-kernels` remains pinned at `6cae49a`.
F is a first-class implementation/adoption family alongside A–E, not an incremental tweak
that must be folded into the incumbent. F0 is the exact donor, F1 a documented minimal
conforming adaptation when required, and F2 a measured hybrid. F-PROJ and F-ATTN are the
first independent comparison tracks; F-MOE and F-LAYER remain separately owned.
Status: registered / untested against our releases. Archived donor scores are not F wins.
Compare complete files and choose the better base; whole-file replacement is allowed.
The dashboard data registers F while retaining the older A–E performance snapshot.
No kernels, validation gates, GPU jobs, releases or scored submissions changed.

## Previous decision: evaluate the complete donor as a replacement baseline

**[2026-10-30: Donor-first adoption — test complete implementations before more invention](2026-10-30-donor-first-adoption.md)**

Own snapshot `5b88889`; donor `alygg77/okc-muse-spark-kernels` at `6cae49a`.
The user correctly challenged treating the better-ranked donor mainly as a list of ideas.
Supersedes the preceding audit's premature restriction against whole-file replacement:
compare the exact complete donor projection and attention files with our actual releases,
then choose the donor, ours, or a measured hybrid. Shared ancestry is not performance parity;
novelty is not an acceptance criterion. Archived projection 1.5502x versus our 1.4976x is a
strong comparison lead, not a promised 3.5% gain. Run independent duplicate controls and
retain all precision, ordering, replay and attribution requirements. Full-file comparisons
are pending; no kernels, gates, GPU runs or scored submissions changed in this report.
Prioritize this adoption decision ahead of new codec/dense rewrites; layer/MoE comparisons
remain separately owned. The next deliverable must identify what code we should use now.

## Previous external-source steering: alygg77 measurement audit

**[2026-10-30: alygg77 public kernels — measured constraints and a narrower next round](2026-10-30-alygg77-public-kernel-audit.md)**

Own snapshot `269c110`; donor `alygg77/okc-muse-spark-kernels` at `6cae49a`.
The supplied X post itself was not retrievable; this is an independently located repository
review, not a quotation of the post. Archived projection 1.5502x is not a fresh rank or a
paired improvement over us. Compare the donor's selected O8/pool8/pool16 routes and router;
query exact four-CTA-cluster occupancy before reopening the failed QKV design. External
measurements distinguish QKV256 transfer pressure from O256's busy mainloop and epilogue,
and document split-K losses despite 94%-peak mainloop throughput. Denser compression stays
CPU-gated, not the sole assumed route to top 100. Attention is unchanged vjeux; KPERM3 merits
a static overlap audit against current MoE work. No kernels, gates, GPU runs or submissions
changed. The report explicitly separates donor reports, our evidence and proposed tests.

## Latest projection report: frontier map and the route to top 100

**[2026-10-30: Projection frontier map and the route to top 100](2026-10-30-projection-top100.md)**

Evidence `c98018f`, check `ad6c673d9a444c1e`. Rank 100 (1.556) needs +4.1% over the
current code's six-receipt median (1.4947), about a third of all headroom to the measured floors.
The qkv one-wave cluster split was built and lost by 9-12 us (parked). The remaining
ranked portfolio sums to 2.7-3.5%, below the target. Only a denser lossless weight code has a
ceiling above it. The m = 256 dense GEMMs hold about 20 us against the 989 TFLOP/s floor
(qkv and o run 590 / 560 TFLOP/s); a batched-GEMM diagnostic is the cheapest test.

Addendum 2026-10-31 (UTC), round 6, checks `cab546f978974515`, `97869fa8302041b5`,
`8b9e8b7c1ecd4c3e`: the codec work model (entropy 10.5 bits; best fixed-rate lossless codes save
4.0-7.1% of the m <= 32 bytes, +1.4 to +2.6% of score if decode stays hidden; one factorial check
decides), the dense whole-tile sketch dropped (identical to the incumbent geometries), qkv_gate
256's loss located in the 72x64 tile's per-k-step efficiency (cuBLASLt split-K large tiles lose
+2.7 to +3.1 us), and router reference toggles worth 0.1-0.2 us at m = 128 only.

## Latest projection decision: corrected top-100 portfolio

**[2026-10-30: Projection top-100 — corrected evidence and structural tests](2026-10-30-projection-top100-portfolio.md)**

Evidence snapshot `c98018f`; verified scored receipt `d85d0081fa8a4a43`, 1.4975856x.
The user's official-CLI cutoff is 1.5560x; the audit did not issue a new CLI query.
P1 Lt enumeration, rt/KLUE router and split exit barriers are shipped; shared-tile P3 lost.
Finish the in-progress QKV split-K, checking actual cluster residency rather than assuming
128 CTAs fit in one wave. Start a bounded raw-BF16 m=256 schedule/epilogue workstream,
not another generic CUTLASS sweep. The report supplies a non-overlapping load budget that
would reach 1.5574x from the user's six-receipt 1.4947x baseline, explicitly a target envelope,
not measured gains. At 800 TF/s only part of the 989-TF/s floor gap is recoverable.
Router/picker cleanup and an offline lossless-code feasibility question remain reserves.
No kernel, gate, GPU run or submission changed. Attention and layer work remain separate.

## Previous standalone decision: attention and projection research roadmaps

**[2026-10-30: Standalone attention and projection — widen the search](2026-10-30-attention-projection-standalone-roadmap.md)**

Evidence inspected through `a48edef` (preceded by `a2bdd7e`). The user explicitly wants better
standings on both component problems, not only layer improvements. For these two owners,
standalone promotion and layer integration are separate gates; the older layer-only priority
must not veto a standalone winner. Current board ranks/quotas were not queried in this audit.

Attention: complete standalone scorecard, then a true cross-tile overlap prototype; resource
co-design and a reduction-ownership challenger are conditional follow-ups. Projection: compare
the actual current portfolio with an independent reference, especially router and high-token
library algorithms; then choose a bounded compressed qkv split-K or lossless WGMMA prototype.
The existing CUTLASS pooler, wide 8-stage picks, DCARRY/PK16/VHOIST and failed local sweeps
are explicitly accounted for. No kernels, validation policies, GPU runs or submissions changed.
This is separate from the layer's M1/tuner-stability work below.

## Latest hourly report, 22:35

**[2026-10-30 22:35 PDT: projection wins and resolved layer attribution](2026-10-30-2235-performance-review.md)**

Snapshot `9bf7f05`. Owned projection C1/C2 win successive paired tests; C1 layer port
is inconclusive. Recovered M-B receipt measures the existing M1 + 51 policy. Mode 79
loses; attention PBULK hangs and its compile fix awaits GPU retest.

## Previous hourly report, 21:26

**[2026-10-30 21:26 PDT: F measurements, combined layer gain and attribution gap](2026-10-30-2126-performance-review.md)**

Evidence `d73c4fd`. Four F tracks measured, with eligibility caveats and measurement-only policy.
Stable M1 + 51 wins a paired layer test; scored gain unresolved. Older pending bundles tested;
hp withdrawn. Modes76-78 miss gates. Missing c8d1 receipt has conflicting candidate attribution.

## Previous hourly report, 14:05

**[2026-10-30 14:05 PDT: Designated finisher and QKV losses; Family F registered](2026-10-30-1405-performance-review.md)**

Evidence `bc10cb1`. Corrected mode 74 loses at 128/256; ownership retained. QKV c4 loses
in tuner observations; paired loop failed. F is registered, untested. Layer outcomes unchanged.

## Previous hourly report, 12:44

**[2026-10-30 12:44 PDT: Projection split exit barrier gains](2026-10-30-1244-performance-review.md)**

Evidence snapshot `c98018f`. Paired router savings at 16/128/256; 8/32 inconclusive.
Release split loses on rt. Standalone score change is inconclusive; layer port/test pending.
A-E standings and pending layer candidates are unchanged.

## Previous hourly report, 05:39

**[2026-10-30 05:39 PDT: Stream-K phase-A retest loses](2026-10-30-0539-performance-review.md)**

Evidence snapshot `15cc6b9`. Mode 72 loses against owned units at 32/128/256 on both
routings. No gate run; layer_hp still awaits GPU tests.

## Previous hourly report, 04:43

**[2026-10-30 04:43 PDT: PRQ win, MoE claiming losses and P3 loss](2026-10-30-0443-performance-review.md)**

Evidence snapshot `b5b8a34`. PRQ31 improves per-request attention; layer_hp is built but untested.
MoE claiming layouts lose even with no claims. Projection P3 fails its stop rule and is parked.

## Previous hourly report, 03:37

**[2026-10-30 03:37 PDT: HMRG standalone win; conforming projection correction](2026-10-30-0337-performance-review.md)**

Evidence snapshot `16573a6`. HMRG merge wins independently of E pipeline experiments and
awaits layer tests. Projection avoids the disputed ordering behaviors; current cross-run
results retain no gain at 32/256. No new whole-layer result.

## Previous hourly report, 02:38

**[2026-10-30 02:38 PDT: E pipeline losses and standalone router gains](2026-10-30-0238-performance-review.md)**

Evidence snapshot `8838272`. New E overlap/residency variants lose cold; earlier E wins remain.
Projection rt/KLUE gains are standalone, with unresolved ordering questions. No new layer result.

## Previous hourly report, 01:31

**[2026-10-30 01:31 PDT: Standalone projection gains; A-E unchanged](2026-10-30-0131-performance-review.md)**

Evidence snapshot `87bc0b8`. Projection Lt/router checks improve standalone components;
the clean candidate reports 1.482x. Two receipts remain absent locally. No layer port or
new A-E result. Standalone and layer promotion remain separate decisions.

## Previous hourly report, 00:32

**[2026-10-30 00:32 PDT: Scored M1 candidates leave gains unresolved](2026-10-30-0032-performance-review.md)**

Evidence snapshot `a48edef`. Two 2.013x scored runs: M1 status unknown in one, off in the
other. Revised census-gate candidates and stable-tuner build await GPU checks.

## Latest layer note: M1's 1x1 gate failure and the route row control (2026-10-30)

**[2026-10-30: M1's 1x1 sliding gate failure in d6c30f88c617438b](2026-10-30-m1-1x1s-gate-failure.md)**:
scored run d6c30f88c617438b (layer_m1rt51, 2.013x) ran with M1 off because its gate failed at
1x1 sliding, where M1 changes no launch (o is cgemv, the front fp32). The one-in-2000
difference has a merge-order rounding signature and mirrors 6ce5be0ba29642ea. The route row
control missed because it copied a zero row (a token with no local route) over a zero row;
fixed and tested offline. After the reset: a targeted reproduction check, then a four-arm
bench of release + mode 51 (layer_g51b) against the release.

## Previous hourly report, 23:35

**[2026-10-29 23:35 PDT: Arm bias leaves mode 51 layer gain unresolved](2026-10-29-2335-performance-review.md)**

Evidence snapshot `dd1777e`. Four-arm mode 51 comparison does not clear its nulls. Runtime
M1 + 51 shows a combined signal; exact enforced-gate M1 + 47 candidate remains unchecked.
Mode-49 attention history diagnostic supports merge-order rounding.

## Previous hourly report, 22:34

**[2026-10-29 22:34 PDT: Mode 51 layer gain and measured M1 limits](2026-10-29-2234-performance-review.md)**

Evidence snapshot `6cb0796`. Mode 51 passes paired layer comparison but is not released.
M1 gains inside its diagnostic build, with little net gain over the release. Attention
input-history dependence remains an untested lead.

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

## Latest layer note: bench arm bias and the four-arm harness (2026-10-30)

**[2026-10-30: Layer bench arm bias, a four-arm harness, and the next comparisons](2026-10-30-layer-bench-arm-bias.md)**:
the three-arm bench's null shared the candidate's state, so an inc-arm penalty (4-6 us at d1
on identical code in 50c4c1328ab742f4 and 720079090ee44cc3, 0.66% on identical-code loads in
29fd004fd58e40ef) passed as a gain. `tools/dl_make_bench_pub.py` now times both files twice
(own prepare each, random prepare and capture order), prints policies side by side and
carry-over tables, and decides on the pooled cross ratio against both nulls. A/A, mode 51 and
M1 runtime bench files are prepared, not run; the +0.84% and +0.87% readings stay unestablished.

## Previous hourly report, 19:31

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

Evidence snapshot `f61d8d9`. B's revisit trigger is met; combined
A/E and projection bundle `b694b4a` is assembled but unjudged. PDL overlap is compile-gated only.

## Previous hourly report, 14:25

**[2026-10-29 14:25 PDT: Attention component gains and validation gaps](2026-10-29-1425-performance-review.md)**

Evidence snapshot `efddc84`. PK16/VHOIST improves 128/256-token attention beyond DCARRY;
the combined package still awaits layer tests. Local commit under the hourly automation's no-push rule.

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

**[2026-10-29 13:27 PDT: Projection context and unchanged family standings](2026-10-29-1327-performance-review.md)**

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
| [2026-10-31 top-10 restart checkpoint](2026-10-31-top10-restart-checkpoint.md) | `8f2c89b`; official board snapshots | Feasibility, current receipt analysis, ranked backlog, tool fixes and exact WSL/CLI resume steps |
| [2026-10-30 Family F registration](2026-10-30-family-f-donor-baseline.md) | ours `7a59655`; donor `6cae49a` | Separate donor-based baseline/hybrid family; F0/F1/F2 variants and independent problem tracks; registered/untested, no adoption or benchmark claim |
| [2026-10-30 donor-first adoption](2026-10-30-donor-first-adoption.md) | ours `5b88889`; donor `6cae49a` | Correct premature restriction on whole-file replacement; exact donor versus release comparison first; choose donor, ours or measured hybrid before new invention |
| [2026-10-30 alygg77 source audit](2026-10-30-alygg77-public-kernel-audit.md) | ours `269c110`; donor `6cae49a` | Scoped external measurements; selective projection routes; cluster residency; dense transfer/epilogue attribution; codec feasibility; attention provenance and KPERM overlap audit |
| [2026-10-30 projection top-100 portfolio](2026-10-30-projection-top100-portfolio.md) | `c98018f`; scored `d85d0081fa8a4a43`; user CLI/frontier review | Corrected target and completed work; QKV residency gate; raw-BF16 m=256 schedule/epilogue test; explicit target envelope versus measured gains |
| [2026-10-30 standalone attention/projection roadmap](2026-10-30-attention-projection-standalone-roadmap.md) | `a48edef`; component sources/logs and public archived references | Independent standalone scorecards and promotion gates; cross-tile attention, resource/reduction challengers; router/Lt portfolio, compressed qkv split-K, lossless WGMMA |
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
