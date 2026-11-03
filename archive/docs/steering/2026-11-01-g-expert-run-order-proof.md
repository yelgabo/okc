# G expert-local unit ordering: feasibility and proof obligations

Source audit only, November1. Main inspected at `3c30f68`; exact incumbent607 lineage is `muse_decode_layer/cand/g1b4_stage_release.py`, public SHA ba9e327c..., pure prefix295ca590..., native CUDA c7c201ae... (pins and owning receipts in `2026-11-01-g-highload-weight-census.md`). No production patch or GPU dispatch. Implementation ownership remains today_moe; this worker supplies only an independent CPU specification.

## Concrete scope

Routing already emits jobs in expert order (`S.jE/jS/jC`, source4219–4249), splitting counts above64 into adjacent jobs of at most64. WGS does not use the non-WGS fat LPT queue. For phaseA/B each job has32/16 output units, with16 K stages per unit. Its flattened schedule is job→outputunit→Kstage. Each repeated job streams the same expert's packed weight units again. Therefore moving whole jobs into expert order is already implemented. The distinct locality idea is transpose within each adjacent same-expert job run: outputunit→job→Kstage. It shortens repeated-weight reuse distance without changing the payload or reduction split.

The exact retained H200 bank `1c6ebc76b91d4b7f` has20active experts/21jobs: expert24 is64+57rows. Its one extra payload is27,525,120B or4.761905% logical requested weight bytes. This is population/source arithmetic, not actual DRAM traffic. Other retained natural banks often have no extra jobs. T32 cannot have>64 selected rows per expert under unique top8. No gain applies there through this mechanism.

## Why a producer-only permutation is incorrect

`wgs_ranges` (6127 onward) partitions a cost-weighted job-stage line, not uniform unit counts. `wgs_wproducer` addresses weights by physical `S.jE[j]`; `wgs_xproducer` (5699 onward) addresses x by physical `S.jrow[j]` and waits on physical `doneAs[j,st]`. `wgs_phase` (6063 onward) chooses `wgs_seg` by physical job ntiles. Every participant must consume the same new logical→physical unit mapping; physical job arrays/packed weights stay unchanged.

Split-unit fixup ownership is currently the CTA containing its first stage: `wgs_cta_of` (6167 onward) inverses the ORIGINAL weighted schedule. After a transpose, calling it with the physical unit's old stage index selects the wrong slot. A valid design needs an exact physical-unit→logical-index inverse, and a new cost prefix using the mapped job's cost (64row and57row jobs can have different ntiles). Producer/consumer sequence and ring parity must advance in logical issue order, while outputs, phaseA releases, `doneAs`, designated finishers and phaseB/expert completion retain physical identities. Tail segments, cyclic `wgs_split`, deferred `WgsPend` flush and per-CTA slot reuse require proof; a permutation bijection alone does not prove progress or fence safety.

Keep all shared proxy fences, WGMMA retirement waits, release/acquire counters and copy counts. Neither changing only weight addresses nor changing only the phase iterator is legal. The minimum honest implementation is a common mapping plus inverse and updated weighted partition model across both producers and consumers, not a one-line launch swizzle.

## Prior closures and reopening decision

- `bfbedf7d48e248a0`: token sort by smallest local expert lost all four six-load comparisons, even excluding sorting cost. It changes input token order; it does not test the above intra-expert output-unit transpose.
- `53c07507b9fd4f8c`: owned8row skinny multicast screen failed broad pairability. G64row jobs and a cache-order permutation are different, but the low overlap warning remains.
- G high-load census row414 in `docs/ruled-out.md` explicitly parks rewriting until H200 L2/DRAM or exposed-stall evidence. Current source audit supplies a different legal work-order specification, not the missing timing evidence. Older interleaved packed layout `0f478ae5371a4b75` changes warp physical layout and is not this scheduler, but cautions against inferring bandwidth gains from contiguous requests alone.

Reopen only if the pending phase attribution identifies meaningful exposed weight work on loads with repeated jobs, and root's six-load sensitivity clears its opportunity gate. Otherwise stop: do not spend a check on this article's historical headline or local census scenario. A repeat only on skewed synthetic routing cannot establish a general score win.

`tools/test_moe_expert_run_order.py`:2CPU tests pass for empty/single/multi-expert runs,32/16 units,64+57 and64+64+1 tails. They prove complete unit/stage bijection, exact inverse and aggregate mapped-job cost, and show singleton runs unchanged. They DO NOT prove native fixup ownership, asynchronous completion progress, graph lifetime or H200 speed. No device prototype is cleared by this test alone.
