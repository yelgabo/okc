# Exact graph-owner artifact STOP: context-wide cleanup can interfere with capture

November 2, 2026. This addendum supersedes the TARGET_HOLD interpretation in [the implementation report](2026-11-02-worker-pr-graph-owner.md) for exact publication **6e454c18c6107f2c4f336a34977292b3c227628a**. It preserves that source/CPU artifact historically; it does not edit the implementation into a second candidate.

The artifact was pushed and all 13 owned files remotely verified at 14:11:08 UTC. Candidate SHA256 **25585f88ec8434023d31d54b78a7184c77959f2b589c870d714a6631b4f595be**, 879911 LF bytes; 13 CPU tests PASS, all seven selected factory dictionaries unchanged. Allocation started 13:41:05 UTC; publication wall elapsed 30 minutes 3 seconds includes source tool/test/HTTP waits, within the 30 ACTIVE-minute allocation. No target readiness or performance qualification was claimed.

## Concrete new contradiction accepted

Immediately after publication, worker read concurrent captain **f3a93a2**, [PR cleanup capture review](2026-11-02-captain-pr-cleanup-capture.md), and accepts its exact source-contract counterexample. This is NOT the previous plain-pointer ownership objection. CUDA graph user objects can retain captured readers as designed, while the separate helper's **cuCtxSynchronize remains context-global**.

Owner A can finish its own host lifetime/eager submissions and queue delayed retirement. An independent owner B can then start capture in the same context. If A's helper resumes, its context-wide synchronization encompasses B's active capture. The documented capture prohibition is not removed by running that synchronization on a separate thread. Thus a valid independent-owner schedule can hit prohibited synchronization/fatal cleanup or interfere with capture. The source has no owner-scoped event completion protocol; the CPU retirement model did not cover this schedule.

This is a source/API contradiction plus a synthetic documented-contract schedule, **not an observed CUDA failure, judge corruption, timing result or disproof of CUDA user objects/compression**. The exact generated host.cpp confirms cuCtxSynchronize and no cuEventRecord. Existing CPU PASS and primary refcount proof do not override it.

## Terminal decision and boundaries

**STOP_EXACT_CONTEXT_SYNC_GRAPH_OWNER_PROTOCOL. No target dispatch, no source/CPU success promotion.** Exact 6e454c1 remains an implemented but correctness-blocked scratch artifact. Public/default ba9, prior plain-owner/serial closures and final private check remain unchanged/reserved. No CUDA compiler, driver/GPU call, challenge CLI/service or check was performed.

Captain's separately queued narrow repair review proposes owner-specific completion records after actual eager submissions plus copy completion, with graph references handling captured readers. That protocol is NOT implemented by 6e454c1 or by this addendum; no synchronization is simply removed and no exception/capture gate is weakened. It requires the distinct queued allocation and new proof/tests before any target discussion. This current original allocation stops on its concrete new contradiction rather than extending into an unnamed readiness loop.

Owned stop-context-sync.json pins this decision and its blocked source. The new stop-publication-manifest.json pins the addendum without rewriting the original implementation manifest. Captain owns README/shared-ledger reconciliation.
