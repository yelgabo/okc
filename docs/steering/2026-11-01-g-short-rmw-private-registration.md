# Private short-RMW whole-layer registration

November 1, 2026. Source epoch follows `468875f`/`ae7818e`, independent ordering
proof `988d618`/`a5cc063` and selected compile proof `9f7d072`. No GPU, official
service call, dispatch or production change. First two reset jobs remain
unchanged; this artifact does not release the held owned-verify job.

## Frozen diagnostic, not a public candidate

`tools/g_short_rmw_check.py::build(root)` emits exact
**f0426f7160768de01997b271744542ae3442d0851a1c27e586ced322a8fc6864**,
**906841 bytes**, `.build/today-moe/short-final-rmw/check.py`. First line is the
layer problem header; second line explicitly says PRIVATE CHECK ONLY, NEVER
SUBMIT. The builder parses/checks the owning607 public receipt: done, correct
board/id/mode,13passed tests, checks.ok, no flags. Its source is pinned to
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.

Control and treatment both retain607's actual main/B4 stage-retirement body.
The only timed treatment change is Tiny/Tiny2's final-counter relaxed GPU
returning RMW, guarded per private owner by a fixed prepare-time boolean.
MID/main/B4 receive no short flag. Resource getters are common, host-only
inspection exports; no active math, native launch, proxy fence or barrier is
changed. Normal `kernel_fn` returns the original607 prepared control state,
with all short flags off. It is not candidate-judged. A later pure candidate
13/12 official qualification and score would be a separate action after a win.

The experiment uses two compiled variants per Tiny key and four independent
mutable placements, not four independent compilations. Control copies0/1
share read-only compiled code; treatment copies2/3 share their distinct code.
All four own scratch, mutable state, nested Lt workspace, inputs/caches,
outputs and captures. Unaffected module objects are deliberately shared code.
Unknown state owners, writable overlaps and unsatisfied memory preflight fail.
The reused clone includes the already-reviewed set ownership fix.

## Fixed decision and observations

- Six actual token loads1/8/16/32/128/256, sliding/full, three banks
  **993811/993817/993821**:36complete rows required.
- Natural routing comes from unchanged FP32-reference inputs. The three banks
  select busiest/second/least EP group by the existing audited method; every
  placement receives the same bank/local_start. Returned IDs must match across
  arms and pass the unchanged routing oracle. No input-dependent timing selector.
-144initial and144fresh-final checks use the actual FP32 layer oracle, cache
  checks and original adaptive thresholds, with added finite-reference/adaptive
  guards. Fresh banks replay8times; only the final fresh replay is explicitly
  observed, not every intermediate replay. Restored and post-timing checks also
  run through the same oracle.
-32short native empty checks: eager plus3captured replays per arm/load; poison
  is inside the captured empty call. These occur before occupied full-model
  banks, without manually resetting native counters. This is explicit
  **empty→occupied** coverage, not a claim of occupied→empty qualification.
- Resource vectors for all five selected module keys and all four owners
  must match exactly and admit at least1CTA/SM. Flags and module cache identity
  are asserted; original compiled WGS/B4 stage body is pinned and a disable flag
  rejected. Runtime resource queries are not achieved occupancy/utilization.
-24cold graph timings per placement/row, randomized placement order,256MiB
  flush. Every event sample must be finite/positive. These are private
  **whole-graph cold GPU-event six-load ratios** using46/16 weights; they are
  not official overhead-adjusted scores or forecasts of board gain. No old
  offsets are transferred and no post-hoc calibration/subset gate is allowed.
- All four control/treatment crosses in each of the three banks must improve
  the full-six factor by>=1% and exceed2xduplicate log-null. Every context/load
  also obeys regression<=max(.5us,1%). Affected-only contributions are reported
  descriptively, not a replacement for the full-six gate. A pass in all banks
  implies the pooled direction; all36raw rows are retained losslessly.
-295s fail-closed budget begins before the original preparation and is checked
  during inner timing/fresh loops and before returning. Private report transport
  is the reused compressed diagnostic capsule, guarded<11000characters; full
  rows/resources/flags/owners/counts are retained, with no truncated partial win.

Original preparation includes native startup checks/plans and may exhaust the
budget. The clock does not interrupt an already-running compiler or CUDA call;
no hard runtime guarantee is claimed. All full-layer preparation and admission,
graph execution and actual H200 performance remain unmeasured here.

## Host/source gates

Two owner tests pass: native literal reversal/resource-only suffixes and exact
source regeneration, plus actual full-score winner/null/regression/missing-row/
nonfinite fixtures. The three preceding source/protocol tests also pass.
Peer reports six actual TorchCPU whole-orchestration tests passing on this exact
artifact, executing generated prepare/owners/flag observer/empty/score/normal
entry and real routing/cache/finite oracles, with explicit native/CUDA arithmetic
stand-ins. Stale outputs/caches, nonfinite outputs/reference, owner workspace
aliasing, wrong flags, failed residency/CUDA, expired budget, invalid timing and
routing permutation reject. Peer will publish the durable fixture/report before
queue clearance; this registration is not its substitute.

Independent source/control/owner review and the final frozen peer commits are
required before root places this after the two existing reset jobs. Stop on
any ambiguity in actual stage/control, missing coverage, relaxed threshold,
unmatched ownership, compile/resource failure, budget or report loss. A losing
complete check closes this exact representation; no automatic flag/fence sweep
or public retry follows. No H200 result or estimated speedup is asserted.
