# Independent owned MODE51 progress-repair review — 2026-11-01

Reviewed owner commit `44e64988254c8998fd91a37591731bd49556cdd0`: `verify_owned_progress.py`, compose and tests. Concrete backend `2b4e0740de26a92fa9ef0f260c28e55f48f9fb2ae8fa20de863e4d8e8718635d`,326791B. Source review found no blocker in the narrowly scoped cooperative+PDLoff+fatal-watchdog repair. This clears neither H200 startup/capture nor performance/public qualification. No GPU/code edits by reviewer.

## Delta and host contract

The generator pins both source/native epochs and exact reused LAUNCH SHA `38026abe7a6f3c70dc36980317c17d62e82e3823f9a78de16246c9e7502b2d28`. It reverses both native edits to prove all remaining native source unchanged. Selected `kernel_fn` sole `_moe1_run` call retains ordinal10 but uses PDLFalse; module name changes to prevent cached old-extension reuse. Compose changes only embedded owned backend plus private source pin, leaving G unchanged. Its production scope is masks4/15, not arbitrary masks.

The reused launcher checks device, SM90/cooperative support, grid/block/shared/chunk/local-expert bounds, checked shared-memory attribute and occupancy, resident>=1 and admitted whole-grid size, then checked cooperative launch with copied unchanged Params on the caller's stream. PDL is rejected rather than silently combined. Exact active kernel function drives occupancy; fatal API failures cannot select the old ordinary launch. Source is ABI/precision preserving. Host CUDA capture/allocator and memory-budget behavior still require H200 validation.

## Active waits and fatal coverage

Native source anchors below refer to generated `_MOE1_SRC`, with a hashed [loop census](evidence/2026-11-01-owned-progress-review/wait-census.json).

| Wait | Producer / ordering | Timeout behavior |
|---|---|---|
| `wfull`/`xfull`, codes2/8 | Bulk completion updates ring full mbarrier; consumers acquire before reads | `mbar_wait_to` calls spin_failed |
| `wempty`/`xempty`,16/64 | Consumers finish reads and arrive ring credits before reuse | same shared helper |
| `xready`,32 | Gather writes, proxy fence+threadfence, gather named barrier then release increments | `spin_ge` calls spin_failed |
| `done_as`,128 | PhaseA publication proxy fence, warpgroup named join, release count; phaseB acquire plus proxy fence before async load | same spin_ge |
| `bq`,256 | Weight producer release-publishes canonical claimed item or negative sentinel; X producer and consumers acquire | bq_get calls spin_failed before timeout sentinel return |
| Final `bc>=G`,4 | Each CTA completes local roles, CTA join, threadfence and atomic arrival; all join before completion-state reuse | direct spin_failed before break |

MODE51 selects ownA/dynB/lean/bar_releaseA, not deferA, a_only or b_only. Diagnostic `mbar_wait_cyc` also calls the helper but is inactive because pa_diag/pb_diag are false for51. Every explicit readiness polling loop reaches the shared fatal helper. Remaining while loops are bounded prefix-search, backward stage search, finite queue traversal and owned-unit count, not publication spins. Native named/CTA barriers, WGMMA waits and griddepcontrol.wait remain intact; fatal helper does not add watchdogs to hardware waits or replace their convergence obligations.

`spin_failed` retains the original clock threshold and diagnostic atomicOr, then executes `asm volatile("trap;" ::: "memory")` before its old return. Thus the caller's old break/return paths cannot continue successfully after expired readiness. PTX8.7 specifies trap aborts execution and signals the host; this is not a value fallback. The counter write need not be visible/recoverable after a failed CUDA context and is not used as permission to continue. [NVIDIA PTX8.7 trap](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#miscellaneous-instructions-trap).

## Progress dependency argument and boundary

Gather publication precedes each CTA's phaseA work. PhaseA depends on gather/rings and does not wait for phaseB completion. Each CTA completes its canonical A units before its consumers move to B; B claims finite unique jobs and waits only for A publications. A fast CTA waiting on another CTA's A credit does not withhold that other CTA's compute role. Local Bqueue publication precedes X/consumer handling, and finite ring reuse follows the original producer/consumer order. Cooperative whole-grid residency removes the missing unscheduled-CTA dependency identified in the preceding audit. No new job mapping, publication count, ownership or cross-phase edge is introduced.

This is a source-level DAG/progress argument conditional on the retained local collective participation and unique-unit coverage; not a machine-checked proof of every original kernel invariant. A latent producer/count bug may still trigger a fatal timeout. The repair makes that failure explicit rather than consuming unpublished data. Existing original memory-order protocol remains unchanged and independently necessary. The final whole-grid arrival precedes completion-state reset; same-state concurrent invocations still need external serialization/private scratch.

With PDLFalse, predecessor completion is supplied by same-stream launch order. Retained griddepcontrol.wait may wait only on prerequisite grids already completed; PTX says it also establishes prerequisite-memory visibility. [NVIDIA PTX8.7 griddepcontrol](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol). No early-dependent overlap is assumed.

## Remaining gate / stop

Owner reports CPU AST reversal/actual launcher mock and local96-register,0-spill,16-barrier compilation. These are owner evidence, not reviewer H200 results. Require exact new artifact qualification covering both selected128/256, both contexts, natural/empty/occupied and fresh private routing states, capture/replay, original math/error bounds, no watchdog/nonzero counters, supported cooperative capture and complete prepare budget. Reject launch/admission/capture errors; a trap is terminal failure, never a retry/fallback success in the same context. No public promotion or runtime claim follows from byte fit, old receipts or this scoped source clearance.
