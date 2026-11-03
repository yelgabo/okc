# Two-stage activation retirement: source-only closure

November1. No GPU dispatch. Root authorized source/compile/progress screen only,
then agreed to source closure after resource evidence. Existing #303 production retained.

## Mechanism and pins

Separate `tools/dl_g_next_expert_pair.py` accepts the exact MOE device from the
#303 stage-retirement control (deviceSHAc7c201ae..., segmentc67020c0...). It does NOT
combine the losing post-stage pending-flush change. Instead it preserves the original
FCI4 publication call and holds consecutive pairs of activation slots until one full
WGMMA drain. After the drain, every reader fences and arrives separately for the previous
and current slot, in producer order. A final odd stage drains/releases alone. Original
outer segment drain remains. Three X slots, all geometry and all math stay fixed.

Pure diagnostic output862343B, SHA256
`cd1f493e2f34a316d90113befc2d4787d2f0909c08a802ca35f831fa85ae7a78`.
Input pure file295ca590... unchanged; generated artifacts are private scratch only.
No production file was modified. `first_seq` resets at every segment so partial ranges,
reordering and mixed A/B continuation retain exact global sequence parity per slot.

## Compile stop

Actual sm90a CUDA12.8 controls are those in the preceding post-stage screen, and
all flags/adapters/manifests are archived. Separate B4FIN flag is the only build difference.
Each kernel entry retains96registers/16barriers. No C7513 warning observed.

|Build|Stack bytes|Spill store/load bytes per entry|Actual LDL/STL across both entries|
|---|---:|---:|---:|
|main stage control|8|4/32|16/2|
|main pair retirement|40|208/504|202/104|
|B4FIN stage control|16|120/216|100/60|
|B4FIN pair retirement|32|172/308|140/86|

Every build has1024HGMMA,512wait1,96wait0 across both entries. Conditional execution
can change dynamic drain counts despite identical static wait0 totals. No dynamic stall
or drain measurement exists, and no benefit is inferred. The new `first_seq` and pair/last
predicates remain live through both unrolled first/subsequent stage bodies, and the first
stage can carry an outstanding WGMMA group into its successor; these are structural sources
of added liveness. Compiler spill evidence establishes the increase, not a precise allocation
of each spill to a variable or register fragment.

The original screening phrase was **materially fewer wait0 and equal resource admission**.
Static instruction census alone does not prove a failure of dynamic-drain reduction, nor does
this CPU build prove fewer admitted CTAs: registers/barriers remain equal and runtime shared
memory unchanged. Closure instead follows the explicitly reported severe spill/stack regression
and root's agreement to stop before GPU. Do not claim failed H200 residency or measured loss.
No parameter sweep or alternate pair-index implementation was attempted.

## Ownership, order and progress model

Both activation descriptors stay valid until wait0 completes every outstanding committed
WGMMA group; after that each reader performs its original proxy fence and release arrival
for each consumed slot. Each slot still gets512arrivals before reuse. Sequence/parity is
unchanged; no barrier is arrived early, twice, or for an unconsumed stage. Register A uses
existing two-way groups with8even commits/stage: after first-stage finalwait1, the preceding
A bank is retired before the next stage overwrites it; nextstageCI0wait1 retires priorCI7
before its bank is reused. Full-drain placement changes no MMA operand values.

Capacity3 supports pair hold2: producer may prefill3slots, so the second stage of a pair
never requires reuse of its first slot. After pair completion all4warpgroups must arrive
before refill. A fast warpgroup can advance to the third prefetched stage and then wait for
the fourth; slow warpgroups can still finish their initial pair from already-ready inputs,
releasing the producer. No reader's local wait requires a slot that it alone must first release.
Pending flush atCI4 remains exactly as control. Same-phase publication order and phase-end
flush are untouched; stage-X/final global joins and original phaseA→B readiness remain.
These are scoped source arguments, not a new proof of the entire upstream scheduler.

Three CPU tests PASS:

- Exact inverse source delta/AST confinement and input rejection.
-2448partial-segment/start-sequence cases, every consumed stage released once,
  one-stage/odd-stage/wrap/parity coverage.
-600random schedules of four independently progressing abstract warpgroups and a
  three-slot producer across multiple segments, all complete with no early overwrite.

`python -m unittest discover -s tools -p test_dl_g_next_expert_pair.py -v`

The finite CPU stage model abstracts actual asynchronous hardware/math/publication latency;
it is not a CUDA proof or a held-out correctness run. Since the compile gate did not justify
GPU, no four-arm artifact or check was prepared. All H200/math/graph claims remain absent.

Evidence `docs/steering/evidence/2026-11-01-g-next-expert-pair/` contains actualcompiler
manifests,logs,SASS opcode census and CUBIN/SASS hashes; rawbinaries remain WSLpersistent
`/root/.cache/okc-g-next-pair{,-b4}/`. Root owns report-index/push integration.

Reopening requires a structurally different representation that avoids the demonstrated
liveness/spill regression, preserved reader ordering/progress proof, and an actual compiler
advantage before a frozen causal benchmark. No more public repeats of the earlier flush placement.
