# GOPT-WS-OWN-05: full-K output-row ownership

Actual pickup: 2026-11-03 06:39 UTC, November 2 23:39 America/Vancouver. General Optimizer read origin/main `55298ac7883795d23d272a6da20ab7dc529ca3db` and accepted the [fixed build brief](2026-11-02-general-hotspot-search-refill.md#post-split-research-decision-full-k-output-row-ownership--november-2-2026-pacific). Four-hour build deadline: 2026-11-03 10:39 UTC. No GPU job has been dispatched.

## Selected design and decision

Use exact standalone release `5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e`, helper `b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716`. One isolated T8/T16 native entry, four output-owning compute warps and one weight producer, 64 rows, K1024 stages, three 36 KiB weight slots, generic activation staging and fused finalization. T1/B4/MAIN and disabled compiler requests remain unchanged. This reorganizes reduction ownership after three helper negatives. It does not retry NS4, norm aggregation or the stopped two-launch split.

The benefit is a hypothesis: remove cross-warp K partial writes, reads and reduction. Weight bytes and MMA counts remain equal; added activation traffic, fewer compute warps and lost early overlap may outweigh it. CPU layout and archived SASS establish source feasibility only. Numerical, device progress, admission, full-call timing and preparation cost remain unproved.

## Execution and gates

Next action is native implementation plus byte-permutation, fragment, publication, lifetime and admission checks. Preserve all finite stops in the brief. A runnable qualified candidate receives at most the authorized private same-process comparison against two independent release controls. Three fresh normal/skewed EP8 banks, all nine correctness sizes, all six scored loads, cold replay, exact source/job identity, input/scratch guards and cumulative bounded partial records are required. Freeze complete-six factor >=1.01, greater than twice duplicate noise for all four crosses in every bank/distribution, no affected 1% regression, and unchanged-load drift <=max(1%, twice null). No topology/shape/ring sweep or automatic retry.

Live outputs stay in ignored `.build/gopt-ws-owned/`; archive after collection ends. No public submission or production promotion is authorized. MoE Worker's direct-input job and CUDA Optimizer's BF16 work remain separate. Independent review is unavailable under the no-new-agents instruction; source, model, compiler and target checks will be identified separately.

## Progress

- Pickup recorded; implementation and qualification underway.
