# Decode32 task inbox

Decode32 Worker owns both contexts of batch 32 decode, T32/q_len1. The user target is a verified **2.3x or better** on `C32 = 46 × sliding_us + 16 × full_us`. Use the owning run's baseline and preserve the other five loads. Research selects architectural experiments. The worker implements, checks, analyzes and integrates them directly. Public submissions require the user's go-ahead. [Ownership](2026-11-02-02-end-to-end-ownership.md) · [Research and execution roles](2026-11-02-04-research-plan-worker-execute.md).

## On wake

1. Read the [job ledger](STATUS.md) and reconcile newer owner entries on `origin/main`.
2. **D32-REDUCE-01** has an accepted private check, `bd5776bdfcab42a7`, for source `93a05524`. The Decode32 Worker owns its single waiter `47403`. Inspect the selected eight split copies, correctness, replay, all twelve cells and the weighted D32 result when the receipt arrives. A promising result needs held-out matched confirmation. [Accepted job](evidence/D32-REDUCE-01/accepted.json) · [Experiment and stop rule](2026-11-02-18-active-reduction-experiment.md).
3. The worker picked up **full-context identity rescaling**. While the sliding check runs, implement and compile the candidate and its disabled control. Verify dispatch isolation, converged voting, resources and preparation budget before considering a target run. CUDA Optimizer owns the B4 phase study. [Research assignment](../2026-11-02-general-hotspot-search-refill.md#2-decode32-exact-identity-rescale-elimination-in-full-decode-attention).

Record actual pickup and exact source/job in [STATUS.md](STATUS.md). For a closed lead, follow the selected-source findings and stop rules before considering new work. [B4 prefix correction](2026-11-02-13-b4-prefix-selected-source.md) · [Observer closure](2026-11-02-17-observer-budget-closure-and-refill.md) · [Standing hotspot goal](../2026-11-02-incumbent-hotspot-goal.md).

[Decode32 report history](../../../archive/docs/steering/decode32/README.md) · [Prior inbox snapshot](https://github.com/yelgabo/okc/blob/0585bd216cb980e8b47a66eef0ba4c3b0c5768a3/docs/steering/decode32/README.md) · [Shared steering](../README.md)
