# Attention continuity: reconcile concurrent implementation, stop duplicate draft

November2,2026. Allocation48111acef2ddccb42e166d06fa5cb32d33b43c76,
incumbent ba9e327c/public607eb52bdaf24160, fresh same-source cc474883117947c0.
This execution first observed the allocation at approximately19:08:48UTC.
It does not reopen parked T1three-launch28248eb5 or pre-gamma56e0c8b.

## Terminal decision and ownership

**Stop the duplicate draft; preserve the already-published canonical treatment.**
Concurrent19ab26cdda573a57db020dfa84decfe4c7fea52b (19:15:48UTC) and
6e1811647d0f525be53a0ebf28bc98e9ca10731e (19:19:35UTC) implemented this same
allocation in another active execution. These commits first became apparent in
this execution's reconciliation around19:23UTC. Do not count the two source
snapshots as independent performance candidates. No further compiler retry,
native integration, tuning or target action followed discovery of that overlap.

The canonical runnable file is
`docs/steering/evidence/2026-11-02-attention-continuity/candidate.py`,
**872908 plain-LF bytes**, SHA256
`e912a5f4378301d62cd978b5012bc5d334dd0894f135784cf0c6cb50c87d6631`.
Its complete native treatment is
`4efb67147807327c18cea0554dd09fa733d0d686c8700c707d047205a528c8e0`;
control is `22c116a45f1817fbaf27a1cc3656380affad49d2cb2367dd23aac83f20b7cef5`.
The native diff, module mapping, build receipts and lifetime inventory remain in
the [canonical report](2026-11-02-worker-attention-continuity.md) and its archive.
This execution did not author or perform those successful four compilations.

Later captain588847c9/fbb68c3f freezes e912 and deprioritizes continuity under the
human step-change/whole-layer ROI priority. Respect that HOLD: no source repair,
harness, GPU, challenge CLI/POST, public/default action or automatic follow-on.
README/index remain captain-owned. Architecture work belongs to its separately
allocated owner, not this reconciliation.

## Independent readback, not another candidate

`tools/attention_continuity_handoff_audit.py` regenerated the canonical file
exactly, checked every file in its archive manifest and independently verified
the four existing source/PTX/CUBIN hashes and corresponding resource receipts.
`published-reconciliation.json` records the exact checked paths and cases.
The unchanged canonical seven-test CPU suite passes again (0.369s).

The audit executes the actual three-definition final `kernel_fn` binding chain,
`_decode_attention7`, `_plan7` and `_launch7` using CPU stands-ins. All12scored
cells plus B33full and three page/fallback cases retain the **complete positional
launch tuple**, including different decode/verify w0/coop bits. Only the four
intended scored cells select continuity. Explicit state scratch owners reach
the workspace allocator; the real CUDA context bind, native computation and
tensor numerics are not executed. Preparation has stand-ins here; the canonical
seven-test suite separately executes its actual preparation function with a
stand-in compiler. This is not a whole-file Torch import or CUDA test.

### Concrete fail-closed scope gap to retain

The canonical final selector correctly chooses control for B33/L1/full/P1.
However, passing that state's prepared continuity module **explicitly** to the
actual `_launch7` with B33 runs continuity: B33 has the same264flat/one-copy/
noncooperative/w0=1024 tuple and the guard checks geometry/flags, not shape.
The executable CPU negative records `direct_entry_rejected=false`. Thus the
stronger claim that every direct entry rejects wrong-shape module use is false.
This is an explicit override-path gap, **not** a default final-binding leak,
an observed GPU corruption, a numerical loss or a proof of unsafe raw-ring
ordering. No repair is bundled into the frozen source; captain owns any later
decision about the direct-entry contract. Do not erase this negative by citing
only the seven existing passing tests.

## Resource/proof boundaries

Reused corresponding canonical L1control/treatment receipts both report128
registers,3barriers,0stack/0spill-store/0spill-load. L8 reports80registers,
16barriers and the same zero stack/spills. This compares each treatment to its
own fresh control, not an absolute-zero eligibility heuristic. These are
secondary NVRTC13/PTXAS13 results, **not this execution's new compilations**.

Canonical L1 dynamic shared allocation grows62464->67072bytes, exactly4608
bytes for64*18FP32 values outside `Smem`; L8 stays114688bytes. With the same
conditional233472-byte SM budget and1024reserved bytes/block used by the
canonical report, two blocks need136192bytes for L1treatment and231424bytes
for L8. The latter leaves2048bytes under that conditional model, not a measured
residency margin. Register allocation granularity, setmaxnreg roles, runtime
occupancy, cooperative admission and actual selected launch remain target gates.

The canonical exact inverse diff preserves all numerical expressions, original
eighteen-value max/rescale/add half-merge, global contribution count, FP8/BF16
boundaries, per-segment page TMA/PGCONT0 and the original fences/waits. Its uniform
post-init join precedes role split; producer final bar.sync2 follows page reads,
and x<ntiles prevents unconsumed speculative page fills at segment reinit.
Shared K/V reuse still needs all original consumer retirements; query/new-row/
merge reuse remains consumer-only. See the canonical writer/last-reader/reuse
inventory, not just a barrier count. Sampled event-model negatives are not an
exhaustive CUDA schedule proof. One detail in the allocation premise needs care:
the selected `-DOKC_W0ENABLE=1` makes the source's `fpart` constant false, so the
old `!fpart` join does execute for these exact flags; the explicit new join is
still present and is not justified by assuming `mbar_fence_init` publishes alone.

Uniform contiguous partitions have at most B-1 cross-request boundaries per
KV-pair partition; this does not measure saved time. Weighted resident placement,
actual nseg distribution, pipeline tail and resource costs must be included.
Most CTAs can pay the cost without cross-request overlap. No profiler activity
sum or the16:46full/sliding weights establish a benefit; current captain ROI HOLD
is binding. Host build/prepare fit, occupancy/admission, fresh outputs/cache,
numerical/replay correctness and full-layer latency remain **NOT_RUN here**.

Captain409757ec subsequently published its completed independent review: the
canonical partition unit test omits active SEGK400/600 virtual gaps. Its new
saved source-witness calculation includes that padding and weighted mapping;
neither synthetic length vector is an observed scored nseg distribution. Reuse
that dated correction rather than presenting this draft's generic interval/
causal event coverage as the exact selected geometry. The captain's disposition
is SOURCE SURVIVOR/TARGET HOLD/LOW-ROI PRIORITY CLOSED, not a measured loss.

## Abandoned local draft receipts

For auditability, `evidence/2026-11-02-worker-attention-continuity/` retains this
execution's simultaneous draft, exact diffs, generator, preregistration and CPU
receipts. Draft7d663524/872403LF used a conditional base-class merge buffer,
not canonical's tail allocation; **it is not the selected implementation**.
Its nine CPU tests pass, including actual final16entry cases, source inversions,
builder/flag/cache isolation and sampled raw/page/init/causal negatives.
Do not transfer those results or its different layout to e912.

One draft control compiler startup at19:22:34UTC failed loading existing
`libnvrtc-builtins.so.13.0` (status7), before PTX/CUBIN emission. The file exists;
the process's library-search environment was not repaired or retried. This is
an environment/startup failure, not a compiler rejection of either treatment.
`compile-result.json`/console and device-prefix receipt are retained. No draft
treatment was compiled and no CUDA driver/GPU/query/install was used.

Final gate: canonical e912 stays frozen/unqualified; duplicate draft stops;
retain the direct-entry negative for captain review. No job exists to wait on,
no invented run-log score, no additional target permission and no new roadmap
assignment. Every owned file is pinned in the adjacent publication manifest.
