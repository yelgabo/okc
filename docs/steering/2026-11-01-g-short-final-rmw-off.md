# OFF prototype: short G helper final-counter RMW

November 1, 2026. Continuation of `d2bdeec` native repair inventory. No GPU,
official API, production modification or public action. This is a new Family G
short-helper representation experiment, not the closed standalone main47 trap
prototype. `607eb52bdaf24160` remains the incumbent.

## Exact source delta

`tools/g_short_final_rmw.py::build(source, enabled=False)` requires incumbent LF
SHA `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
It requires helper0 literal139768B SHA
`b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716`.
The patched literal is SHA
`9193a5e0aac01fdfba4c609f2b3400d9eead63bbd0949eaf09ddac05c350aec4`.

Only the final `bc=p.ctr+4+(S.gen&1)` RMW receives a compile-time alternative:
`atom.acq_rel.gpu.global.add.s32` becomes explicit
`atom.relaxed.gpu.global.add.u32`, retaining volatile inline assembly, its
return register and memory clobber. Counts are nonnegative and bounded by the
existing checked grid, so signed/unsigned addition has identical bits here.
No unused-return CUDA atomic/RED lowering assumption is made.

The default has no opt-in define. Explicit enabled generation appends
`-DOKC_SHORT_FINAL_RELAXED=1` only to `MOE_TINY_DEFINES` and
`MOE_TINY2_DEFINES`. An enabled native compilation fails if not Tiny, not WS,
SPLIT or WGS. M16/helper0df, B4FIN32 and main128/256 receive no flag and keep
their old acq_rel final counter. The distinct WGS `fc` RMW is unchanged.

Generated scratch artifacts (not production):

|variant|LF bytes|SHA256|
|---|---:|---|
|OFF|869371|ad7c5936e55fb4ff44cec644e1c37ea7feb89e0b6c6a4a25f6cec9c64d789aa3|
|explicit opt-in|869551|d33915c7e8579d815a0adf8a182d685570185e8c6c1b5fd6c036854a5fbcdd44|

Manifests are attached under `evidence/2026-11-01-g-short-final-rmw`.
The generated source is reproducible from the tracked incumbent; no source
encoding or runtime file reads are added. No full qualifier has been built.

## Independent ordering assessment

The auditor identified helper0 native lines1510–1583: all helper lanes finish
their global epilogue writes, execute their own `__threadfence`, then join
`WS_JOIN` over NTL threads. Compute threads converge at their matching join
(~2405), then CTA_SYNC over NT (~2426). Thread0 retains its GPU fence, performs
the final counter RMW, polls with GPU acquire for the full grid count, then
all compute threads cross the retained post-CTA join before final reads.

This reviewer independently read [PTX8.7 §§8.8–8.9](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model).
A fence preceding a strong write forms a release pattern; observation order
can traverse atomic RMWs. Thus each CTA's fenced publication reaches the final
GPU-acquire poll through the same-location strong GPU RMW chain. CTA joins
collect and redistribute the cross-thread visibility. This argument depends on
the actual retained writer fences and synchronization, not just the leader's
fence or an assumption that relaxed means release by itself. Explicit atomic
read/write lowering must be verified in PTX. The scope/proxy/location conditions
are unchanged. Atomicity and per-location consistency establish the counter
chain; they do not replace the release/acquire visibility argument.

All CTAs capture S.gen before their arrival; the opposite parity counter is
reset before the next epoch uses it; the last arrival advances generation.
The current counter is not cleared while readers poll. Ordinary same-stream
completion prevents a subsequent invocation racing final reads/reset.
All reset/generation text is unchanged. Checked cooperative admission and
PDLoff remain; this delta introduces no new polling or dependency edge.
The selected full-grid readers are generic finalization loads, not permission
to drop any generic/async proxy bridge elsewhere.

## Local validation and release gate

Three CPU tests pass using the actual source: normalization of only the one
CUDA literal plus the two optional final definitions leaves the complete AST
identical; all fence/join/reset/wait tokens and distinct fc operations remain;
counter epoch arithmetic is checked over arrival permutations up to7CTAs.
That last fixture is arithmetic, explicitly not a PTX memory-model simulator.
Source drift is rejected. Run:

`OKC_REPO_ROOT=ROOT python -m unittest discover -s tools -p test_g_short_final_rmw.py -v`

Independent actual sm90a Tiny/Tiny2 PTX/SASS/resource checks are pending at
initial report publication. They must show the exact GPU fence→strong relaxed
GPU returning-RMW→GPU acquire→CTA join chain, unchanged math/barrier/proxy
instructions and no worse resources/admission. OFF must retain the control
atomic. No forecast is made from a single atomic representation change.

If these gates fail, stop at source/compile and preserve the incumbent. If they
pass, a future root-authorized private same-weight four-owner comparison still
requires all six workloads, heldout natural routing, strict finite/.008 global/
.075 row math, fresh captured inputs, empty→occupied without manual reset,
generation/replay checks, resource/admission and preparation/transport bounds.
Require >=1% whole-six, >2x duplicate spread, each seed and per-load regression
gates. No GPU or public qualification is implied by this source report.
