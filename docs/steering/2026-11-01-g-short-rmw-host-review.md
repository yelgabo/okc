# Independent host review: private G short-counter comparison

2026-11-01. Reviewer base `f46f79c`. Frozen owner draft
`.build/today-moe/short-final-rmw/check.py`,906841 bytes, SHA256
`f0426f7160768de01997b271744542ae3442d0851a1c27e586ced322a8fc6864`.
No source edits to the owner's worktree, GPU execution, official/service call
or public action. This is private-check host support, not a release candidate.

**No remaining host blocker found in the reviewed scope. Six actual Torch CPU
tests pass in2.648s.** They execute the generated owner/prepare/score/empty and
normal-dispatch functions, actual per-owner native flag wrapper, and actual
routing/finite/adaptive/cache validators. CUDA/native/reference arithmetic are
explicit stand-ins; this is not H200, CUDA host-link or full-model math proof.

## Source and ownership reviewed

Owner builder `tools/g_short_rmw_check.py` uses exact607e stage-on source and
`g_short_final_rmw.py`, preserving the earlier local-compiler proof. Only the
two short module definitions receive the new RMW flag when `_srm_on` is true.
Normal judged preparation/entry stays control, and returns the original fully
prepared production state. Existing WGS stage-retire source remains literal;
there is no required STAGE_RETIRE=1 flag in the baseline. The runtime denies
an explicit stage-disable flag. No older stage-off comparison is reused.

There are **two compiled variants per short key** (control and treatment),
with **four independent mutable placements**, not four independent compiled
binaries. Duplicate controls share their immutable native module, as do
duplicate treatments. Unaffected mid/B4/main modules are shared and their
resources/policy remain common. The actual clone functions check private
scratch, graph inputs/output storage, attention buffers and Lt workspace
ownership while permitting inventoried immutable weight sharing.

The five native resource keys are queried for all four placements; exact
vectors and admission are compared, and selected flags/module identity are
asserted. The CPU facade tests the policy and failure handling; the actual
H200 getters, values, admission and allocations remain unexecuted. Prior local
Tiny/Tiny2 PTX/SASS proof is in `2026-11-01-g-short-final-rmw-compile.md`,
including the post-counter mutable `.cg` load census. Ordering proof is the
separate `2026-11-01-g-ordering-restoration-proof.md` (988d618/a5cc063).

## Actual executed CPU host fixture

`tools/proj_g_short_rmw_host_test.py` imports the previously reviewed reset
state facade, then executes the **current generated** suffix and validators.
State construction/native arithmetic/FP32 model references are CPU stand-ins;
cache writes, routing checks and numerical error checks use real Torch CPU
tensors. CUDA graph/event execution is modeled explicitly. Empty graph
recording includes output poison and the native zero operation, so a missing
captured native op is rejected rather than passing on a stale zero.

The positive whole-prepare fixture completes36 rows: all six loads, sliding/
full, three fixed seeds993811/993817/993821, four owners. It observes:

- 144 initial model validations and144 fresh-final validations in the report;
 actual validator invocation count576 also includes restored/post-timing checks.
- 144 model captures. Each bank gets eight repeated fresh replays and only
 their final state is observed; intermediate replay correctness is not asserted.
- 32 empty component checks before the normal banks: eager plus three observed
 poisoned graph replays for each short size/owner. This tests **empty to
 occupied**, not occupied to empty. There is no manual scratch/counter reset.
- Original production state returned with treatment disabled; all six normal
 dispatches use control. A direct bad-flag test exercises the real wrapper.
- Full report transport below11KB, including a realistic varied-time fixture.
 Equal timing stand-ins produce no gain; a synthetic winner and a single-case
 regression exercise the actual full-six-load/all-cross/per-seed decision.

Negative tests reject stale model replay, corrupt cache, stale empty graph,
nonfinite reference/adaptive calibration and output, shared scratch owner,
wrong compiled flags, nonresident resources, unequal valid routing order,
fatal CUDA synchronization errors, initial and inner fresh-loop deadline
expiry, NaN event time and zero summary time. TF32 and the reference cache
are restored/cleared in finally on success and failures.

The native math stand-in does not implement expert arithmetic, real FP8
cache conversion, TMA/barrier progress or CUDA stream capture. The real
reference validators use a small CPU arithmetic model instead of4096/8192
weight GEMMs; no error tolerance or production reference was changed.

## Gate disposition

Independent **host-scaffold clearance only** for this exact private artifact.
Keep formal source proof, local compiler evidence and CPU orchestration
evidence distinct. A new artifact needs its delta reconciled and relevant
fixtures rerun. Root must still assess realistic295s preparation/memory cost,
actual native routing/resources and H200 math/fresh/empty checks, then require
the preregistered full-six-load gain beyond independent nulls with regression
protection. No performance decision follows from CPU timings or source size.

The official service is unavailable and the earlier accepted layer job has
unknown outcome; this review authorizes no retry, substitute run or dispatch.
