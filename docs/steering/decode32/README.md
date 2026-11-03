# Decode32 task inbox

## Priority override: Decode32 is P4

The user explicitly re-ranked Decode32 **P4**, not the project's P0 or exclusive performance focus. [Standing goal and top-100 cutoff gaps](../2026-11-02-incumbent-hotspot-goal.md#latest-user-priority-override--november-2-2026-pacific). Continue the worker's already-owned accepted identity-rescale job (`4a2a79bdac04499a` at last indexed status) and own result analysis/correctness without interruption; don't abandon accepted work because of a scheduling change. **No new Decode32 experiment gets automatic P0 preference.** Central steering prioritizes standalone MoE W4A16 and BF16 projection refills first, while Decode32 retains end-to-end T32 ownership. Historic D32 2.3x target remains a performance objective, not overall allocation priority.


Decode32 Worker owns both contexts of batch 32 decode, T32/q_len1. The user target is a verified **2.3x or better** on `C32 = 46 × sliding_us + 16 × full_us`. Use the owning run's baseline and preserve the other five loads. Research selects architectural experiments. The worker implements, checks, analyzes and integrates them directly. Public submissions require the user's go-ahead. [Ownership](2026-11-02-02-end-to-end-ownership.md) · [Research and execution roles](2026-11-02-04-research-plan-worker-execute.md).

## On wake

1. Read the [job ledger](STATUS.md) and reconcile newer owner entries on `origin/main`.
2. **D32-REDUCE-01** job `bd5776bdfcab42a7` is terminal and archived.13/12/12 pass, D32 C32=12.305490ms/1.814995x. Stop this exact eight-copy performance treatment; no useful affected-path signal or routing correction is established. Waiter47403 ended. [Receipt and decision](2026-11-02-19-reduction-result-identity-next.md).
3. **D32-IDENTITY-01** is implemented with enabled/disabled twins, converged vote, unchanged resource admission and eleven passing CPU/source/host/reader tests. Own bounded private job `4a2a79bdac04499a` is accepted for paired source `f7a5d1fb`, published `59011284`. Follow its sole waiter and preserve partial evidence on failure. Target cost/correctness/performance remain open. CUDA Optimizer owns the B4 phase study. [Frozen gates and source](2026-11-02-19-reduction-result-identity-next.md) · [Research assignment](../2026-11-02-general-hotspot-search-refill.md#2-decode32-exact-identity-rescale-elimination-in-full-decode-attention).

Record actual pickup and exact source/job in [STATUS.md](STATUS.md). For a closed lead, follow the selected-source findings and stop rules before considering new work. [B4 prefix correction](2026-11-02-13-b4-prefix-selected-source.md) · [Observer closure](2026-11-02-17-observer-budget-closure-and-refill.md) · [Standing hotspot goal](../2026-11-02-incumbent-hotspot-goal.md).

[Decode32 report history](../../../archive/docs/steering/decode32/README.md) · [Prior inbox snapshot](https://github.com/yelgabo/okc/blob/0585bd216cb980e8b47a66eef0ba4c3b0c5768a3/docs/steering/decode32/README.md) · [Shared steering](../README.md)

[Driver monitor: observer closure, completed joint research and independently executed owner checks](../2026-11-02-driver-monitor-0330.md). Historical planning references do not override any accepted source or owning protocol.
