# Native G per-stage observed-readiness diagnostic

November1. Root dispatched the frozen private standalone MoE artifact as
**f147d06a64504c1e**. Root owns status, terminal receipt and profile archive. No public
submission or dispatch by this worker. No H200 measurements are claimed in this checkpoint.

## Frozen bytes and backend

Artifact private `.build/today-moe/g-stage-check.py`,392222B, SHA256
`bf0ca4ae44281d30efd80b988566dd8c431f3095a1d38e4310029c39667f8170`.
Builder `tools/moe_make_g_stage_probe.py`; device transform `tools/dl_g_stage_probe.py`;
body `tools/moe_g_stage_body.py`; CPU analyzer `tools/dl_g_stage_analyze.py`;
receipt reader `tools/moe_read_g_stage_probe.py`.

Uses the exact #303 native stage-retirement MOE_CUDA_SRC body and host launcher,
SHA256c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71,
from tracked ba9e327c startup-wrapped layer. The standalone harness keeps256expert
weights and real global IDs, selects the same local32 read-only weight slice, and retains
existing controlled BF16/SC6 packing. Actual TorchCPU bytes matched root layer versus
standalone packers for W13/W2, SC6on/off, plus code/scale/fat/skinny layouts; evidence
packing.json. Scratch Python allocation is at least the exact G host's required extent
(Python WGS_XROWS4160 versus native3776); indexing comes from the exact native host.
No packing, math, fence, barrier, retirement, launch or route-order optimization is tested.

**MAIN128/256 only. T32 real B4FIN is explicitly unmeasured.** The prior standalone
MAIN32 comparison cannot silently stand in for #303's B4FIN finalizer. More B4 binaries
risked the300second setup limit, so they were not built into this check. Ordinary judged
entry remains the uninstrumented native backend. This is attribution, never a speedup candidate.

## What can and cannot be learned

No software timestamp directly reports autonomous TMA completion. Readiness is observed
at entry or original wait-return; serial weight-then-X waits attribute only the latency
exposed by that order. Two sparse builds separate:

- READY: read-only nonblocking acquire test of weight/X full barriers at entry, then
  timestamp before original weight wait, after it, and after original X wait.
- RETIRE: timestamp before existing stagewait0, after it and after the original per-reader
  proxy fence and activation-empty arrival. Existing separate segment drain remains.

First/last MMA hooks were removed after the initial all-edge observer distorted the binary.
They are NOT measured or inferred as exact issue times. No producer observer, extra blocking
wait, release change or extra CTA/warp synchronization exists. All original MMA and wait1/0
operations remain. Submicrosecond windows include observation/instruction overhead and are
not unbiased exact stalls. Snapshot flags precede the entry tick slightly.

Only consumer thread0 is sampled, CTAs0/G/2/Gâˆ’1, phaseA/B ordinals0/7/15/31 when actually
present. Whole phase ranges, actual seqstart/end, job/expert/start/count tables, kernel entry
and post-consumer-CTA-join timestamps are recorded. Every expected selected ordinal must
exist; absent ordinals must remain zero. Analyzer reconstructs real unit/stage/segment/job
from original trailing-partial-first phaseA order and phaseB order, not from an assumed
uniform partition. Capsule preserves local per-token masks; it is an ownership trace,
not a complete numerical-input replay capsule.

## Low-overhead storage and ordering proof

`S.fcnt[0..2]` is dead in the selected WGS/NW16/!TINY/FAST1off build. Original accesses
are only the excluded tiny FAST1 split path. At phase entry thread0 stores its own trace
rowbase as two32bit words and seqorigin there; only that SAME thread reads them later,
using scoped inlinePTX volatile shared loads. No interthread publication is needed. Smem
size, all original field offsets and ring layout are unchanged; no allocator alias is added.
Shared pointer words avoid an alignment assumption. Clock/predicate/pointer temporaries
stay inside each inlinePTX block, not in C++ state across WGMMA groups. Phase metadata
is written outside the accumulator loop. Trace is a private fifth scratch tensor (475int64),
with an empty unused fourth debug placeholder. Host validates device/dtype/shape/extent;
Params layout is compiled consistently in the same CUDA TU. Every decoded replay zeros
its trace before execution, including high32 readiness bits and unused rows.

Entry test_wait is nonblocking and does not arrive or consume any barrier. Original
full-slot waits, all512reader retirement arrivals, writing/reading proxy fences, WGMMA
commit/waits, fixup publication and cooperative grid joins remain unchanged. Marker stores
address only the disjoint trace allocation, never model/scratch values. The three original
mutable scratch tensors, every output and every trace have unique storage per arm.

Independent reviewer today_proj checked shared deadness, bounds, PTX/compiler ordering,
fifthscratch/Params/graph/flagmapping and final host logic. Final exact bf0ca4ae cleared;
reviewer independently ran the5actualTorchCPU host tests. Their device checkpoint b6f043c
and review note are coordinator-owned. H200 host compilation/capture/admission remain gates.

## Static compiler evidence and rejected observations

All four FINAL source variants share sourceSHA
`311f874cc70cd69c3028556f42f2c5f2098296af57bb2f8711cc60b8b4b85402`.
Actual CUDA12.8 sm90a CUBIN census, across both false/true entries: every variant has
1024HGMMA,512wait1,96wait0. The disabled observer's full opcode census equals the
original #303 native control census. No C751x warning in the final variants.

|Build|Registers|Barriers|Stack/entry|Spill store/load bytes/entry|
|---|---:|---:|---:|---:|
|Uninstrumented|96|16|8|4/32|
|Endpoint-only|96|16|8|4/32|
|READY|96|16|8|44/68|
|RETIRE|96|16|8|4/32|

These are local NVRTC/PTXAS static properties, not H200 runtime attrs/latency. H200 must
retain actual same geometry and achieved occupancy; localBytes may differ and is printed.
An earlier endpoint manifest was built before a correction in an uninstantiated marker;
it was rerun from the identical FINAL source, so archived final receipts are consistent.

Rejected CPU-only iterations: full all-edge/2WG probe MAIN200/684,B4FIN220/968spillbytes
plus C7519 and static HGMMA duplication; splitting C++ helpers still READY136/244 or180/372,
RETIRE204/456 or156/344; scoped PTX+sharedpointer eliminated the damaging retained liveness.
No rejected source was sent to H200 and no timing conclusion came from those attempts.
The first PTX conditional-assembly spelling failed locally and was replaced by a constant
predicate branch. Final source compiled; no GPU quota was spent on compiler syntax repair.

## Six arms and frozen interpretation gates

C0/C1 truly uninstrumented duplicates; E0/E1 endpoint-only instrumented references;
READY and RETIRE detail arms. Six independent scratch/output placements, shared readonly
weights and fixed input pointers. Four binaries. All base GPU preparation completes before
extra extensions compile on two CPU threads; no GPU tuner or math runs concurrently.

Held-out normal global routing: M128/256, seeds76111/76117/76123 (+M passed to the existing
public-distribution generator).25rotated cold rounds per arm. Three separately cleared/cold
snapshots per instrumented arm; selection is median phase span, never speedup or best stall.
Private baseline math checks original/fresh, valid all-local skew, exact-zero remote, restored
input. Tallies normal72,fresh36,skew36,remote36,restored36, plus every decoded snapshot.
Threshold global.008,row.075; output AND reference finite and scalar errors finite. No relaxation.

Freeze per load/seed/detail mode:

- Duplicate whole-kernel spread<=1.5%; endpoint-only whole displacement<=1.5% versus GM C0/C1.
- Detail whole-kernel displacement<=3% versus GM C0/C1.
- Detail selected phase spans<=3% versus corresponding GM E0/E1; endpoint phase spread<=1.5%.
- Same actual shared/threads/residency and no increased registers; all rows complete and
  actual metadata/sequence agrees with the replayed input. Errors are fatal; timing gate
  failures retain raw evidence but forbid that row's bottleneck interpretation.

No matched stable row => no bottleneck conclusion. Passing rows still do not establish a
candidate speedup, all-stage roof, HBM/L2 transactions or an exact submicrosecond stall.
275s inner/final guards and270s build/case guards fail closed; they cannot interrupt a slow
native compile or already-running kernel. Compressed binary capsule+compact complete six-row
summary is guarded below11000bytes. Actual receipt transport remains an H200 runtime gate.

## Pre-measurement phase opportunity table

607e current46/16-weighted layer/expert envelope from medians/separate profiles:

|Load/backend|Layer us|Expert us|PhaseA startup/steady/retire|PhaseB startup/steady/retire|
|---|---:|---:|---|---|
|D32 B4FIN|191.738710|74.987097|unmeasured|unmeasured|
|V16 MAIN|269.346774|161.183871|pending this check|pending this check|
|V32 MAIN|359.298387|204.412903|pending this check|pending this check|

The old1c6ebc76b91d4b7f fixed-owner finalizer bound and duplicate-byte census do not fill
these cells. Prior stage-retirement's1.85â€“2.87% component gain is ALREADY in607e and gets
zero new credit. For the two measured MAIN loads only, equal fully exposed9.052106us
saving on both, or equal5.03870% expert reduction, is required for1% whole-six-load score
with four other loads frozen. A future phase-deletion or sampled-cadence extrapolation is
an explicitly optimistic transfer scenario, not a measured bound; do not add overlapping
phase windows or multiply sparse waits into a fabricated all-stage roof. Real B4 transfer
needs its own measurement. Root will add the runtime table only from qualified rows.

## Tests and handoff

3source/mapping/completeness tests PASS;5actualTorchCPU host tests PASS, including actual
wholeprepare/25x6timing scaffold, realistic retained-route metadata, 64bit timestamps,
allformatter/capsule transport, stale-fresh output, NaNreference, budget and gate negatives.
Actual pack byte equality gate PASS. CPUfacade is host testing, not simulated CUDA proof.
Reproduce with hostvalidatePython fromtools:
`python -m unittest test_dl_g_stage_host -v` and
`python -m unittest discover -s tools -p test_dl_g_stage_probe.py -v`.

Evidence finalcompiler manifests/logs/census and packing:
`docs/steering/evidence/2026-11-01-g-stage-probe/`. Raw CUBIN/SASS in persistent WSL
`/root/.cache/okc-g-stage-{control-final,endpoint-final,ready-v4,retire-v4}/`.

Root owns f147d06a64504c1e wait/archive and report-index/publication. The new locality paste's
expert-band/unit-major question will be evaluated only after this table gives genuine new
headroom; the prior local proportional+.628% duplicate-byte scenario alone is insufficient.

## Terminal addendum — f147d06a64504c1e, 2026-11-01 05:52 PDT

Source evidence commit `5c1a1f8`; reviewed artifact bf0ca4ae44281d30efd80b988566dd8c431f3095a1d38e4310029c39667f8170 unchanged.
H200 PASS5, uninstrumented official control 2.3402777977x, CI [2.305173631,2.375381965],
prepare171.324839s. This off-board standalone score is not a layer score or candidate improvement.
Root owns canonical receipt/profile archive for this exact ID; derived complete decoded capsule is
`evidence/2026-11-01-g-stage-probe/f147d06a64504c1e.decoded.json`.
All registered math coverage passed: normal72, fresh36, skew36, remote36, restored36;
worst global/row error .0060922033/.0071880231 versus .008/.075. Every arm used96registers,
231168B dynamic shared,576threads,1resident CTA; READY alone reports8B local, others0.

**All12 detail mode/case rows fail the frozen attribution gates. No rerun.**

|M / seed|Control GM us|READY whole / phase displacement|RETIRE whole / phase displacement|Endpoint phase duplicate spread|Qualified detail|
|---|---:|---:|---:|---:|---|
|128 / 76111|185.792|5.615% / 10.825%|4.909% / 10.256%|1.968%|neither|
|128 / 76117|185.518|5.374% / 10.894%|5.426% / 11.209%|2.483%|neither|
|128 / 76123|184.783|6.243% / 13.671%|5.568% / 11.315%|3.081%|neither|
|256 / 76111|202.448|9.539% / 14.327%|8.670% / 13.042%|5.325%|neither|
|256 / 76117|214.512|7.019% / 12.118%|5.825% / 11.455%|2.718%|neither|
|256 / 76123|221.583|8.644% / 14.903%|7.835% / 12.672%|1.252%|neither|

Control/endpoint whole duplicate spreads .259–.918% and endpoint whole displacement .043–.485%
are stable; detail whole displacement4.91–9.54% exceeds3% in every case. Detail phase displacement
10.26–14.90% also exceeds3%. Five of six endpoint phase duplicate spreads exceed1.5%.
Thus observed startup/steady readiness and retirement numbers remain raw perturbed observations,
not a bottleneck diagnosis, floor, or opportunity bound. Neither the median positive waits nor
sparse ready-at-entry counts establish bandwidth limitation. Correctness and unchanged static
HGMMA counts do not rescue failed distortion gates.

### Updated incumbent opportunity table and decision

|607 incumbent load|Layer / expert us|A startup / steady / retire|B startup / steady / retire|New attributable savings|
|---|---:|---|---|---|
|D32 B4FIN|191.738710 /74.987097|unmeasured|unmeasured|not established|
|V16 MAIN|269.346774 /161.183871|detail probe rejected|detail probe rejected|not established|
|V32 MAIN|359.298387 /204.412903|detail probe rejected|detail probe rejected|not established|

Standalone held-out generator inputs have real global256 IDs and real expert weights, but are not
actual layer router outputs. Their phase proportions cannot be substituted into607 layer timings.
The already measured retirement gain is excluded. Equal9.052106us exposed saving on BOTH MAIN
loads (or equal5.03870% expert reduction) still supplies only a conditional1% six-load target.
There is no measured startup/cadence/retire decomposition to credit toward it after this check.

### New retained H200 route population, independent of perturbed timing

Counts/job metadata are valid replay input observations even though clocks fail attribution.
SC6 payload per unique expert is27,525,120B: W13=18,350,080B, W2=9,175,040B.
Extra jobs repeat both phases' logical weight requests. They are NOT measured L2 misses/HBM bytes.

|M / seed|Local routes|Active experts /64-row jobs|Extra jobs|Extra W13 / W2 MB|Extra/request payload|
|---|---:|---:|---:|---:|---:|
|128 /76111|170|22 /23|1|18.350080 /9.175040|4.3478%|
|256 /76111|332|22 /24|2|36.700160 /18.350080|8.3333%|
|128 /76117|191|22 /23|1|18.350080 /9.175040|4.3478%|
|256 /76117|307|26 /26|0|0.000000 /0.000000|0.0000%|
|128 /76123|246|21 /22|1|18.350080 /9.175040|4.5455%|
|256 /76123|336|27 /27|0|0.000000 /0.000000|0.0000%|

Conditional scenario: delete that request fraction of the ENTIRE expert latency, hold the other
four layer loads fixed, and apply those two new standalone populations to607 layer envelopes.
Paired seed six-load factors are 1.012567288, 1.004403492, 1.004606903;
geometric mean 1.007185400. This is an optimistic proportional scenario,
NOT a timing bound, cache model, measured saving, or forecast for actual layer routing.
Compared with the earlier local64-bank+.628% scenario, this retained H200 three-pair scenario
is about+.72%, with only one pair exceeding1%. New evidence proves duplicate-job population
at M128 too (max85/90/122); M256 max150 gives two extra jobs in one bank, none in the other two.

Stop: close this exact observer, no additional GPU diagnostic. Park expert-band reordering:
its bijection/ragged cost/fixup proof remains necessary but is insufficient without measured
exposed traffic or robust score headroom. No source candidate, release or public action follows.
A future reopening requires low-distortion hardware traffic/cadence evidence or a representation
change with a stronger independently validated score-covered opportunity; no generic sweep.

Resource qualification: 1 CTA/SM is the runtime occupancy/admission limit from the CUDA occupancy API, not measured achieved occupancy or utilization.
