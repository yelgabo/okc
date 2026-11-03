# Independent WGS stage-end X retirement review

November1; parent root evidence `e1c3727a812211264bf9dbc070ac19241071ba40`.
Scope is exactly `.build/today/dqscreen/stage-x.cu` SHA256
**a231faabd8b1613096b6288d3c7a77811c5378b8b7476c375d711eb9e1b38eef**,
versus repaired ctrl.cu c5ececf50cb8893ca6189cda42a051228b4a8576faf71142cccfa0b5917c3fa4.
No H200 run or production edit by this reviewer. An earlier alternative was parked
at root steering; this review does not propose another release mechanism.

**Verdict: no new ordering/lifetime/progress blocker in this exact delta.** It
strengthens retirement before each X release and retains the all-reader release
protocol. This is a source/memory-model review, not GPU qualification or a speed claim.

## Actual compiled evidence

Independently counted actual disassembly sections for both lowered moe_fused
specializations; counts below are **per entry**, static instructions:

|Source|HGMMA|wait_group1 lowered waits|wait_group0 lowered waits|
|---|---:|---:|---:|
|G0-grid, MFIN off|512|256|16|
|Repaired ctrl|512|0|512|
|Exact stage-x|512|256|48|

Thus stage-x totals1024 HGMMA /512 wait1 /96 wait0 across both entries, versus
ctrl1024/0/1024. The evidence is actual WARPGROUP.DEPBAR.LE operands, not merely
absence of a C75xx warning. All were compiled with the recorded sm90a/CUDA12.8
MAIN flags, PRMTSC1/SC6W1/DQ7=1. Stage-x each entry uses96 registers,16 barriers,
8B stack,4B spill stores and32B spill loads. Launch bounds remain576threads/min1CTA.
The nonzero stack/spills are real and must be included in the runtime/performance
gate; restored wait overlap alone does not establish a win or zero-local contract.

Root-produced compiler manifests/logs and independent counts are archived under
`muse_decode_layer/profiles/stage_x_independent.2026-11-01/`. The reviewer did not
modify the parent's source or claim a new compilation of the unchanged twin.

## Exact scope and completion proof

The only algorithmic delta is in wgs_seg2095–2207: remove prev and its next-stage
release; after all current-stage commit groups, execute aligned `wg_wait<0>()`,
the same per-reader shared proxy fence and one mbar_arrive1 on **current xs**;
then increment seq. Remove the old final prev arrival, retaining the outer wait0.
Weight release, all full-slot waits, WGMMA groups/wait1, producer code, expected
counts, ring sizes, routing, math, epilogues and global joins remain byte-identical.

The [PTX8.7 WGMMA wait rule](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
makes wait0 complete every preceding committed group. Therefore all reads using
the current X descriptor finish before its release. The wait executes uniformly
across each consumer warpgroup. Each of512 consumers then fences and performs a
release arrival; the [mbarrier acquire/release ordering](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait)
prevents the producer's following refill from preceding those readers. Existing
proxy fences remain on every reader; this argument does not substitute a leader's
fence for other lanes. See also the [proxy-fence definition](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).

The producer first completed BX_F/TMA delivery; only then can the consumer issue
the current WGMMA reads. Its BX_E pending count stays NT512, and the same512
arrivals release each used slot exactly once. Releasing at current-stage end is
earlier than the old next-stage first-group point, but is safe because the new
explicit full retirement precedes it. No subsequent segment epilogue reads this X
slot: it uses completed d registers, global fixup/epilogue buffers and unchanged
publication state. Weight data already copied into registers uses its unchanged
separate BW_E protocol; the change does not relax A-register reuse waits.

## Edges, lifetime and progress

- A nonempty wgs_phase interval has0<=s0<s1<=16. wgs_seg executes the first stage,
  then s0+1..s1-1, producing precisely s1-s0 X arrivals per consumer. A one-stage
  segment releases once; its outer wait0 does not duplicate the arrival.
- A zero-job/empty phase breaks before wgs_seg, so issues no new X wait, WGMMA or
  release. The existing zero-job metadata guard and all routing initialization
  remain unchanged. No fictitious prev=-1 slot is referenced by the new code.
- Phase-A trailing-partial-first order, split-unit boundaries and phase B retain
  the same stage visit order. seq advances once per stage across segments/phases;
  slot=seq%3 and parity=(seq/3)&1 still match both producers exactly. Across ring
  wraps, producer reuse still requires all512 previous-phase arrivals.
- Faster warpgroups cannot cause early overwrite: their arrivals alone cannot
  finish BX_E until the slowest reader's group retires and every lane arrives.
  The added drain waits only on already issued operations whose current operands
  were ready. It introduces no dependency on a future stage, producer refill or
  another CTA, so creates no new circular wait. Existing cooperative grid/global
  publication and scheduling assumptions are retained, not newly proved here.

Three CPU gates pass in `tools/test_dl_wgs_stage_retire.py`: exact inverse delta;
49794 empty/partial/full/reordered interval cases across both phases with unique
stage visitation and512 arrivals per used slot; and actual SASS/resource assertions.
The model checks ownership/counts, not CUDA execution or hardware scheduling.

No memory-order primitive is removed dynamically for any released stage. H200
correctness/fresh/empty/stress gates and actual native occupancy/local-memory
resource admission remain mandatory before promotion. Do not silently relax a
zero-local guard because this variant spills. This proof is specific to the pinned
current-stage full-retire delta; later drain removal, stage-count or reader-count
changes require a new review.
