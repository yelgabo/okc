# dense handoffs

## 1. gmb_entry: glue boundary reductions without cluster.sync, and its PDL trigger at entry (2026-11-04 18:15 PST)

- Patch tool: `python3 tools/dl_dense_gmbe_patch.py [--base F] [--out F]` (default eat_best.py -> work_dense/gmb_entry.py).
  It applies the edits of `tools/dl_dense_gmb_patch.py` and then inserts a small rewrite of the two glue PDL8 edit
  lists. Anchors are code only: the glue helpers `cluster_squares` / `cluster_squares_pair`, their call sites (6 / 1 / 1
  occurrences, including the anchors inside `_PDL8_EDITS` and `_PDL8_GRAFT_GLUE_EDITS`), the two `__shared__ float
  partials[...]` lines, and the line `_PDL8_GRAFT_GLUE_EDITS = [`. Every edit asserts its count.
- Base: 920587d5 (best_3). Output: work_dense/gmb_entry.py, SHA256
  1880297e7eaf4c7bb68711d0cca3cf38ca29ad08bdf05538577b8315e5b965d2, 918,725 bytes (cap headroom 81,275).
- What changes, at T = 1 / 8 / 16 only (b1 = `glue_cluster_boundary`; the T = 16 b2, T = 32 boundaries and every
  T > 32 path are untouched, see `_graft_boundary` / `_pdl8_graft_boundary`):
  1. Each of the two cross-CTA sums of squares sends its per-warp partials with `st.async ... mbarrier::complete_tx`
     into every cluster CTA's shared memory and waits on its own mbarrier, instead of DSMEM stores + `cluster.sync()`.
     Same values in the same slots, same butterfly order, so b1's outputs are the base's bit for bit.
  2. The glue kernel's `griddepcontrol.launch_dependents` moves from just before the first reduction to the kernel's
     first statement, so the fused pooler + router launches during o / b1's pre-wait part. Alone this is null; with
     change 1 it is needed, because the shorter b1 otherwise left the pooler's pre-wait prologue and weight stages
     late (pooler critical path +0.5 to +1.3 us at T <= 8 with change 1 alone).
- Reference check: PASS 12/12 (work_dense/box/ge/chk.log), native MoE build, guards pass; every A/B run PASS too.
- Paired file A/B (tools/dense_ab.sh = solo_ab.sh, c/w alternating, 60 runs per process, eval settings), two series
  pooled (work_dense/box/ge, ge2), candidate minus the mean of its two neighbouring controls:

| cell | delta us | pairs faster |
|---|---:|---:|
| decode 1 sliding | -0.62 | 8/8 |
| decode 1 full | -0.73 | 8/8 |
| verify 1 sliding | -0.66 | 7/8 |
| verify 1 full | -0.68 | 7/8 |
| decode 16 sliding | -0.45 | 8/8 |
| decode 16 full | -0.67 | 8/8 |

  Expected score effect from the touched cells: about +0.33% (delta times % of score per us from the pool table).
- Untouched cells (decode 32, verify 16 / 32; the changed kernels never run there): ge2 gave +0.47 to +0.79 us at five
  of six with 0-1/4 pairs faster; ge3 (3 more pairs plus a third arm, base + trigger only, which also runs identical
  kernels at these loads) gave the candidate -0.17 to +1.35 and the third arm -0.81 to +2.80 (verify 32 sliding +2.80,
  0/3). So per-file shifts of this size appear at cells whose kernels are identical; I read them as the process-level
  layout effect the solo run saw with bfq (decode 1 +0.13 to +0.18), not as a cost of this change. Please re-measure
  them on your side; if they persist across independent builds, that is a separate problem worth a look.
- Per-kernel evidence (tools/dense_ccode.py condition A, 2 rounds x 10 replays per file; work_dense/box/kc, kt):
  b1 1.61 to 2.04 us vs 2.16 to 2.74 in the base; the pooler's critical path back to the base's or below
  (decode 1 sliding 14.80 vs 14.95).

### Happens-before argument (synchronization change)

- Writers and readers. In each reduction, lane 0 of each of the CTA's 4 warps writes one float (two for the pair
  reduction) into slot `rank*4+warp` of every cluster CTA's `partials` array; all 128 threads of every CTA then read
  all 32 slots. No other writes touch these arrays.
- Release. Each write is `st.async.shared::cluster.mbarrier::complete_tx::bytes.b32` aimed at the destination CTA's
  mbarrier `gmb[k]`. Per the PTX ISA (st.async), the store is performed and then the complete-tx on the destination
  mbarrier is signalled, with release semantics at cluster scope. (Source: my reading of the PTX ISA st.async
  section; this session could not re-open that section, because the online ISA page is truncated before chapter 9.7.10.
  The compiler accepted `try_wait.parity.acquire.cluster` and the SASS shows SYNCS.PHASECHK.TRANS64.TRYWAIT. Please
  check the st.async memory-semantics sentence against the ISA before shipping.)
- Acquire. Every thread of the destination CTA spins on `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64`
  for phase 0 of `gmb[k]`. The phase completes only when its pending arrival count is 0 and its tx-count is 0. Thread 0
  did the single expected arrival with `arrive.expect_tx` of exactly the bytes the CTA receives (32 x 4 B, or 64 x 4 B
  for the pair), right after `mbarrier.init` (count 1). So the phase completes exactly when all 8 CTAs' stores into this
  CTA have landed, and the acquire makes them visible to the reading thread. A complete-tx that arrives before
  the local expect-tx cannot complete the phase, because the pending arrival is still outstanding.
- Initialization. `mbarrier.init` and `arrive.expect_tx` run before the kernel's existing pre-wait `cluster.sync()`,
  followed by `fence.mbarrier_init.release.cluster`. That `cluster.sync()` (barrier.cluster arrive.release / wait.acquire)
  orders every CTA's initialization before any CTA's first st.async, which comes after `griddepcontrol.wait`, after
  that barrier.
- Two reductions use two mbarriers (gmb[0], gmb[1]) and disjoint arrays, so no slot is rewritten and no phase is reused
  within a launch. Shared memory and mbarriers are fresh in every launch.
- Exit. A CTA reads and exits only after its own wait has seen every byte addressed to it, so no DSMEM store
  targets an exited CTA.
- Progress / co-residency. The 8 CTAs of a cluster are launched and resident together (cluster launch guarantee), and
  every CTA executes both send loops unconditionally before its wait, so the unbounded waits always complete; there
  is no capped poll.
- Fences kept. The removed `cluster.sync()` calls existed only for this exchange (they followed the DSMEM stores
  and preceded the reads). The pre-wait `cluster.sync()`, `griddepcontrol.wait` and every global store / load
  ordering of the kernel are unchanged. The trigger move changes only when the dependent grid may launch: the pooler
  still returns from its own `griddepcontrol.wait` only after this grid has completed and flushed, as before.

### Integrator review of entry 1 (18:20 PST)

ACCEPTED, pending two evals. Rebuilt with your tool on best_3: identical SHA 1880297e. Diff read against every rule:
no fence or release/acquire removed beyond the two exchange-only cluster.sync calls; the byte counts match the
senders; init is fenced before the pre-wait cluster.sync; each mbarrier is used once per launch (no loop in either
glue kernel); the waits are unbounded and cluster co-residency holds; the entry trigger only lets the dependent launch
early. PTX ISA st.async: the store is a weak operation and its complete_tx on the mbarrier has release semantics at
cluster scope, as your argument says. My own A/B (c/w x3, 60 runs, all loads, check PASS 12/12): d1 sl -0.66 (3/3),
d1 full -0.64 (3/3), d16 sl -0.74 (3/3), d16 full -0.96 (2/3), v1 sl -0.45 (3/3), v1 full -0.29 (2/3); untouched
cells mixed (d32 -0.37 / -0.19, v16 sl -1.85 / full +0.67, v32 sl +0.28 / full -1.54), no consistent direction.
