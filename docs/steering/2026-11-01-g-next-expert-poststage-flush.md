# Family G next expert candidate: post-first-stage pending publication

November1, source/CPU/compiler checkpoint only. No H200 job or public action.
Parent owns qualification/publication. Independent review requested from today_proj.

## Decision and exact delta

Implement one change on the exact current stage-retirement control. Move the pending
previous-unit `wgs_flush<PB>(p,S,pd)` from the first stage's commit-group index4 to
immediately after `stage(std::true_type{})` returns. The first stage already performs
its full WGMMA drain, per-reader shared proxy fence and current activation-slot arrival.
All those operations remain. The flush is before subsequent stages and the current
segment epilogue, which may install a new pending record. Phase-end flush remains.

Actual source sets `OKC_WGSFCI=4`, `OKC_WGSGS=2`. The old adjacent comment mentions
an earlier CI3 scout; it does not describe the compiled value. This is distinct from
root's old pre-stage flush on the serialized control, and from outlining the segment.
No SC6, decoder, tile, ring, math, precision, routing, launch or finalizer knob changes.

Input pure layer SHA256 `295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491`.
Actual MOE_CUDA_SRC SHA `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`.
Segment pin `c67020c073a65b7042104903c1f679fbd7230c23d8bcd0e76e69306631340be3`.
Generator `tools/dl_g_next_expert_flush.py`, API `patch_source(source)->str`.
It rejects unpinned device input and verifies all Python AST outside that CUDA literal unchanged.
Generated `.build/today-moe/g-next-flush.py`: **861728 bytes**, SHA256
`8195660dec57d9ceb6d6ab71d048fd5c144e657d8ec44022c75b0f34ca88bca1`.
This pure file does not yet include the production startup wrapper; compose the identical
reviewed validator separately and re-pin those exact bytes before release.

## Actual opportunity, without projecting a gain

Public stage job `607eb52bdaf24160` passed13 at2.126082676, CI2.094191436–2.157973917.
The steering snapshot reports rank303 and rank50 at2.2894: another7.68% score required.
The earlier stage-X paired job `63f5a2a8ed2341d1` passed its gate at1.85–2.87% six-load;
that validates that scheduling mattered, not this new placement.

From the current 607e medians/profiles, 46/16 weighted layer/expert microseconds:

|Load|Layer median|Profiled expert|
|---|---:|---:|
|decode32|191.738710|74.987097|
|verify16|269.346774|161.183871|
|verify32|359.298387|204.412903|

Profile timings and benchmark medians are different samples. Under the explicit frozen
other-times assumption and treating equal fractional expert savings as exposed critical-path
savings, a1% whole-six-load score gain requires3.79184% shorter experts across these three
loads. Hypothetical5% shorter experts would contribute1.3251% score. These are sensitivity
requirements, not expected speedups. No measured pending-flush or spill-time attribution exists.
The source still streams SC6 weights35840B/stage,32A+16B units/job,16stages/unit;
activation traffic/ring capacities are unchanged. Only compiler-local spills change statically.
No changed HBM-request or measured-bandwidth claim is made.

## Actual CUDA12.8 sm90a compiled result

Both false/true entries were compiled for each build using root's pinned
`tools/local_cuda128_compile.py`, SC6=1, forward-declaration adapter, ordinary MAIN
flags and then the same flags plus `-DOKC_B4FIN=1`. No CUDA driver/GPU execution.
Host torch wrapper is excluded by this local compiler. Virtual headers and exact flags
are in each archived manifest. All four builds complete; no C7513 warning observed.

|Build|Registers|Stack bytes|Spill store/load bytes|Actual LDL/STL across both entries|
|---|---:|---:|---:|---:|
|main control|96|8|4/32|16/2|
|main post-stage flush|96|0|0/0|0/0|
|B4FIN control|96|16|120/216|100/60|
|B4FIN post-stage flush|96|8|8/36|16/4|

Every entry retains16barriers. Actual nvdisasm across both entries in every build has
1024HGMMA,512wait1 and96wait0; pipelining is preserved. These are static unrolled counts,
not dynamic stall duration. Evidence manifests/logs and SASS opcode census+CUBIN/SASS
hashes: `docs/steering/evidence/2026-11-01-g-next-expert/`.
Raw reproducible CUBIN/SASS retained WSL `/root/.cache/okc-g-next-{control,flush,control-b4,flush-b4}/`.

## Publication/lifetime/progress argument

The previous record's epilogue stores/reductions precede the next segment exactly as
before. `wgs_flush` body is byte-identical, including each lane's global proxy bridge,
warpgroup joins, release/acquire atomics, fixup reads/zeroing, and phase-A doneAs publication.
Moving the call later cannot publish before those writes. Per-reader activation retirement
uses the identical completed-current-stage drain/fence/arrival sequence reviewed for stage-X;
no later flush reads the released activation slot (it reads registers, global fixup and metadata).
Weight-ring release is unchanged. Shared `S` job tables/lifetimes remain valid through the
entire segment. Previous `pd` is consumed and cleared before any current epilogue overwrites it.

Publication order among records within a consumer warpgroup is unchanged. The newly inserted
work before publication is first-stage compute: phaseA reads already-gathered input, phaseB
reads phaseA results. It does not await the previous record's same-phase publication. In phaseA,
DF is off and fixup completion uses the existing nonblocking acq_rel count. In phaseB the
existing designated finisher can wait for other shares; the added first-stage compute depends
only on phaseA publication, not a phaseB finalizer. All phaseA consumers retain their phase-end
flush before proceeding to their phaseB consumer work. Thus this placement adds no new
same-phase publication dependency/circular wait. It does not repair or re-prove unrelated
upstream scheduler assumptions. Cooperative admission and all existing phase joins remain.

Activation release now occurs before the delayed flush: no consumer keeps an X slot while
waiting inside that flush. Producer ring progress can therefore continue to the existing
next-stage capacity, never requiring a future refill to complete the already-issued first stage.
Empty phases never enter a segment. One-stage segments still flush old `pd` before their own
epilogue, then install/flush the new record by existing logic. Partial/reordered visit order
and final phase flush are unchanged. No watchdog/fall-through or fence relaxation is introduced.

Existing proof anchor: PTX8.7 WGMMA wait completion and the prior reviewed
[stage-X retirement proof](2026-11-01-wgs-stage-x-retire-review.md). Atomics/proxy operations
inside the unchanged flush keep the same documented publication pattern.

## Tests, acceptance and stop

`python -m unittest discover -s tools -p test_dl_g_next_expert_flush.py -v`:3PASS.
Exact-source/AST scope test, unchanged flush body/primitive census,3264 pending-kind,
partial-stage/parity/ring fixtures, and publication-relative-order model. These are CPU
structural models, not hardware correctness or an exhaustive inter-CTA scheduler model.
Independent review remains required before root dispatch.

If root measures, require identical private startup validator, native identity, all official
13cases and12cells, finite/fresh/mixed/remote/guard gates; resource failure is fatal.
No relaxed tolerances. Compare complete six-load timings, count the three unchanged loads
as zero opportunity, and reject any affected-load regression beyond controls. A causal paired
follow-up should use3held-out inputs and duplicate arms, every crossing>1% six-load AND2xnull;
current quota/public single-arm measurement cannot supply that causal proof on its own.
A public result must be described as a single-arm measurement with independent routing noise.

Stop on any safety/correctness/resource failure, prepare>=300s, serialization returning,
or no meaningful whole-score evidence. Delaying flush may lose useful overlap even though
spills fall. Do not promise an improvement, retuneFCI, remove drains, or claim rank50 fromstatic
counts. No new tuning sweep is authorized by this candidate.
