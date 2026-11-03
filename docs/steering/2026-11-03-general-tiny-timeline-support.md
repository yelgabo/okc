# Tiny timeline support

Actual pickup: 2026-11-03 07:31 UTC, shared source59f06a8f. The final-day directive supersedes the earlier refill. General takes source-bound event/coverage validation and bounded failure-record support for Tiny/Tiny2. The MoE owner retains native observer integration, GPU dispatch, its sole collector and any resulting optimization. This is an explicit split of the support work, with no second producer or GPU job.

The previously accepted row-owned7f10243c98e74746 is complete and remains stopped. The newer conditional row-owned dispatch wording does not reopen it. No NS4, row/warp/K/ring sweep or T32+ engineering is included.

Selected source is standalone5df257af/helperb8ed8543, not a nearby WGS literal or the new row-owned kernel. T1 uses Tiny; T8 uses Tiny2. The support deliverable will bind actual build flags and native event anchors, check per-CTA range/unit coverage and monotonic timestamps, retain partial records within a fixed byte budget, and require duplicate-control observer-cost validation before any phase claim. Existing layer-level traces do not measure these native Tiny events.

No target timeline or exposed bubble is established at pickup. No run is allocated here. Source review and CPU tests will distinguish sample points, globally published data and critical-path hypotheses. The owner remains responsible for target admission, correctness, fresh replay, full endpoint timing and final-day acceptance gates.

## Completed source and CPU support

Checked the owner's published observer at source05e3e875, payload106a05c54e2ffd8b0f851b142e5d8ef3f74a47e92112d09d145c2017d637ee81, observer helper8324f079, and its registration6485937b. The owner's sole job is f4012a0531af4013 with waiter95489. This support neither dispatches nor polls it, changes its frozen source, adds a gate retroactively, or edits its tools.

The [validator](../../tools/gopt_tiny_trace_contract.py) binds thirteen actual source points and exact Tiny/Tiny2 flags. [Seven independent CPU tests](evidence/2026-11-03-general-tiny-support/cpu-tests.txt) pass. They use synthetic timestamps and test occupied/empty routing, all 132 CTA ranges, per-job B readiness coverage, chronological streams, the selected WSBL tail split, missing/corrupt exported witnesses, frozen observer-cost arithmetic and lossless size failure. The actual owner's decoder and export format are exercised. These are source/schema checks, not a second native build or a target timing review.

The maximum tested short-load assignment is188 units per CTA, within192 slots. Tiny flags hand the final two B epilogues to compute warps, Tiny2 hands over one. The raw-buffer audit checks the exact routing-derived ranges, those tail flags, serial helper records and one readiness wait at every B job transition. The exported-case audit reconstructs the two retained unit streams, checks delta coordinates and unit counts, and recomputes which sampled finishers may be selected. The owner's existing receipt reader checks summary rows but does not fully revalidate those retained unit details; this checker supplies that additional pass on completed exported evidence.

The tested owner-format report with irregular, target-scale synthetic timestamp differences occupies7540 base85 characters. It contains both T1/T8 traces, all 132 CTA summaries per case and the two retained unit streams, with no detail compaction. The full raw-buffer JSON alternative occupies312295 characters for the T8 stress fixture and fails the12000 bound. This rejects that tested export layout, not the owner's bounded summary, every possible codec, or the instrument. The [summary and fixture](evidence/2026-11-03-general-tiny-support/summary.json) record the scope. No timing comes from an H200.

## Interpretation limits for the existing accepted check

- The B-end and finish stamps belong to compute thread 0. They do not mark completion of every compute warp or every output store. Tiny's second retained tail belongs to warp 1. The later join and grid protocol remain the relevant completion boundaries. The retained CTAs are the last two sampled thread 0 finishers, not proven last physical CTA exits.
- The exported unit interval named compute includes decode/MMA work, partial-buffer backpressure and the existing consumer barrier. It is not isolated MMA time. Helper epilogue sample points are distinct from publication; the final helper sample is before its thread fence. Existing doneA publication and the original grid barrier establish data readiness.
- A ring-wait endpoint records when the consumer passed its wait. The accepted observer does not stamp the actual FAST1 TMA issue. Startup or first-ready minus entry therefore includes routing and other work, and cannot be called pure first-fill latency.
- The tested source already streams B weights across the A/B boundary. A proposal to add cross-phase prefetch needs a specific observed missing dependency, rather than assuming no prefetch exists.
- The existing observer's frozen cost screen allows max 2 us/10 percent intrusion and bounds duplicate-owner differences by max 0.5 us/2 percent. Passing it supports exploratory timing interpretation, not a 2 us correction forecast. Costs, concurrent CTA intervals and phase shares cannot be summed into a causal gain.

These findings do not invalidate the active check or authorize a replacement. The current observer already documents several of these limits. This support makes the source distinctions and independent exported-data checks explicit. Independent native reviewer/target access is unavailable here; no new agent or GPU job was created.

## Handoff and next gate

When the owning collector ends and its analysis is published, run `audit_exported_case(tokens, item)` from this validator on each item in the verified reader's `report.traces`. Keep the reader's owning source/job association, exact failed/complete status, unrounded owner validity and frozen cost gate. CPU schema success always returns `phase_gain_qualified=False`. A missing witness or failed cost/budget/order gate remains a missing or invalid diagnostic, without a retry. General will inspect shared published evidence on a later wake, without starting another collector.

Any correction remains with the final-day owner and needs the complete short-load candidate test: a consistent 2 us or larger T1/T8 saving under the frozen multi-bank/duplicate-noise gates, other four loads inside noise, correctness/replay, and confirmation. The stopped row-owned design stays closed. This helper does not create a new kernel candidate, quota allocation or public permission.

Fresh public snapshots and the promotion guard were refreshed before closing. The current public MoE row is e859eddb768c4194 at 5751.372090677 us/2.338311587x, 95% CI[2.303236913,2.373386261],PASS5/6/6,return0/no flags. This is another owner's public run, not a result of this CPU support. Its server payload identifier is0b9bf6df3123a7b856a1a7fb1a1345123d40f51f244cf0924301efd99102a430; correspondence to physical release bytes is not established here. Public source attribution remains with the release owner. The guard finds 17 numerical private leads and zero disposition gaps against the refreshed boards. [Public snapshot/receipt and guard evidence](evidence/2026-11-03-general-tiny-support/manifest.json).
