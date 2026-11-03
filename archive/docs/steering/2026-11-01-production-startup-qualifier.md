# Single-arm repaired hybrid startup qualifier

2026-11-01. Source/CPU review complete for the file below. **H200 host
compilation, resource admission, runtime correctness, prepare time and
performance remain unmeasured.** Root owns any authorized official action;
this worker made no GPU/official call.

Frozen `.build/today-proj/prod-size/qualify-mask15.py`: **980787 bytes**,
SHA256 `4ad326c3451d1b9a7ba22e6c89a2b06ebd8f565c20300abd9fc0d3b0a35001bc`.
The file has 19,212 bytes spare under the strict one-megabyte cap. It is
readable source, without compression, encoded kernels, runtime source-file
reads, comparative timings, extra comparison arms or new algorithm selection.

## Exact source epoch and timed behavior

The base is the separately repaired full-mask15 core, 952309 bytes,
`81f056f5e495f5cab0a66183a5b56cab783e7fa47732ed58e0ffa4feed5127a1`.
Its owned backend is `2b4e0740de26a92fa9ef0f260c28e55f48f9fb2ae8fa20de863e4d8e8718635d`.
Progress repair `44e6498` and independent source review `e57dd21` cover the
cooperative expert launch, PDLFalse and fatal timeout trap. This is not the
earlier ordinary-launch 951539-byte epoch.

`prod_size_qualify.py` changes only the owned host query/declaration/export
and module-cache epoch, then appends prepare-only helpers. Both `_MOE1_SRC`
and `_ONE_SRC` device values are asserted identical to the repaired core.
The G source and the sole normal `kernel_fn` AST remain unchanged. All four
verify16/32 sliding/full overrides are fixed; other eight cells and unsupported
metadata keep the original G delegation. No startup result chooses a kernel.

The normal production prepare runs once, retaining G's original startup
validator and 56 observed expert replays and the ordinary owned prepare.
Then a second **private owned prepare** creates fresh scratch, norm objects,
WD diagnostic/runner and Lt closures/workspaces. This is one candidate's
private validation owner, not another timed or comparative arm. A shallow
or generic deepcopy would share mutable Lt closure workspaces and is not used.
Tensor/module storage and the three Lt closure workspaces are checked against
production ownership. Production G is not run during these private target
checks: every call first requires the actual positive override bit.

The exact original production state is returned. No private input, oracle,
capture or output is installed in it. G's existing native/guard state is
not cloned or mutated by the target validation. Compiled code and immutable
input weights may be shared. TF32 changes are bracketed around the new FP32
helpers only and restored in `finally`; `_AB_DEQ` is cleared on entry/exit.
The outer 295-second guard starts before the original production prepare.

## Full-model and empty-component coverage

For each of four target cells and two preregistered seeds (938101,938123),
generate the published natural input distribution, then select its busiest
reference expert group before copying data and update both reference local
starts. Fresh input is independently generated with seed+101. Real routing
remains natural; no full-model expert IDs are injected or fixed.

Each bank runs two private-stream eager checks, joins that stream, then
captures the unchanged public kernel path. The initial replay is observed
before inspection. Three further observed replays alternate fresh/base/fresh
at the captured pointers. Outputs and newly written KV cache slots are
poisoned, then all input metadata/cache contents are restored as appropriate.
Output poison precedes input restore to permit a legitimate output/input
alias. References use the unmodified base/fresh cache bank, not a cache after
the candidate's in-place writes.

Total: **48 full-model validations, 8 captures and 24 changed-input replays**.
There are two distinct input banks per seed, not 24 independently generated
fresh banks. Every validation checks finite front/back/output values, natural
routing acceptance and nonempty local expert work, hidden/residual updates,
actual new cache slots and zero WD/expert diagnostics.

Separately, each case exercises actual owned `_moe1_run` on the same private
scratch with legal unique global IDs32..39, local start0, BF16 `[M,4096]`
inputs, FP32 weights1/8, variant10 and PDLFalse. All IDs are remote to the
selected local32 group. This avoids the invalid assumption that local_start256
means empty (it aliases0 under modulo256 routing). Exact zero output and
zero diagnostics are required eagerly and after capture/replay, including
output poison. Empty replay is followed by an occupied full-model replay
without resetting expert counters: **40 empty checks, 8 component captures
and 32 restored-occupied observations**, separate from the model totals.
Timed input dispatch is not changed to create these component cases.

## Oracle and resource contract

Twelve registered FP32 model/input/check helpers are copied by AST at
generation time from `dl_make_bench_pub.py` BODY, hash
`54881e3c51e58af323021e439e8ea3dd8c63bb2db7b7dbb184ca60c22adc9e4a`.
PRELUDE hash is `50a546330f1a0f3ad98d0d564b42858c1b960311b7e38880dfbce07b17b56f8f`;
copied helper hash is `2d4a1fbd192bc26d7e22434d50a65d1ca1342688f6857c025eac3b69adc9eb65`.
The oracle is official-style FP32 arithmetic, including the unrounded
attention result; it is not relabeled as SGLang's BF16 stage oracle.

The independent model, FP32 router/routing and cache tolerances/formulas are
unchanged. A checker clone adds only fatal finite checks around adaptive
calibration/relative metrics; a back-reference wrapper checks both BF16-query
and FP8-query reference outputs. This prevents a nonfinite FP8 calibration
from being silently clamped to a normal tolerance by Python min/max.
An AST reverse test proves that removing those added guards restores exactly
the original checker. No bound or arithmetic expression is relaxed.

New prepare-only APIs report actual CUDA attributes for MAIN51 and
ONE8/mode3, ONE8/mode51. Each vector contains registers, local bytes, static
shared bytes, dynamic shared bytes, threads, max threads, blocks/SM, SM count,
cooperative support and resident limit. Calls to SetAttribute/GetAttributes/
device attributes/occupancy are checked. They require zero local bytes,
valid opt-in extent/thread limits, positive admission/cooperative support and
consistent cached ONE residency. The cooperative expert launcher still
checks its actual full-grid admission on every launch. Diagnostics, resources,
CUDA execution and math failures are fatal; none triggers a fallback.

Legal unsupported weight packing is different: the existing production
weight preflight skips owned preparation and delegates to G. The qualifier
prints selected0/unsupported_weights explicitly. Such a receipt cannot be
treated as four-cell candidate qualification. Required success prints
selected4/48/8/24 plus empty40/8/restored32 and all three resource vectors.

## Review, limits and handoff

Independent source audit `13e70c4` clears this exact4ad326c3 file: original
G/timed path/device bytes, source provenance, selectors for all twelve cells
and metadata negatives, oracle bounds, private ownership and host API scope.
MoE's separate actual Torch CPU facade covers the real orchestration with
explicit native/CUDA/reference stand-ins, routing/cache checkers and negatives
for stale results, nonfinite output/reference, ownership/workspace aliases,
bad routing/cache/resources/diagnostics/deadlines and fatal CUDA errors.
These are CPU host contracts, not a GPU execution result.

Final peer host evidence is published as `d18dff2`: seven tests pass on the
actual frozen4ad artifact, including whole preparation, real metadata/oracle
checks with explicit arithmetic/CUDA stand-ins and resource-getter mocks.
This adds evidence only; no artifact/source change follows the review.

Owner `prod_size_qualify_test.py` passes two additional tests: the exact
checker reverse proof and actual getter C++ compiled/executed with isolated
Zig CPU CUDA mocks. Normal resource vectors and all18 API error positions
are exercised; errors propagate fatally. Host mocks do not link Torch/CUDA.
No broader testing or new kernel changes were added after these gates passed.

The global 295-second guard is a required runtime limit, not evidence that
G plus two owned preparations and all oracles fit. Actual H200 host linking,
local-byte values, cooperative capture/admission, numerical gates, empty/
restore progress and full official correctness must still pass. Root may
decide whether to perform the single-arm public measurement under the user's
standing authorization. This file contains no causal comparison benchmark;
private paired evidence and actual whole-six-load score remain separate.
No speedup or top-10/top-50 feasibility claim is made here.

Reproduce using the exact repaired source (tracked compose tool regenerates
it from the prior comment-packed core):

```text
python tools/prod_size_qualify.py ../okc REPAIRED_MASK15_SOURCE .build/today-proj/prod-size/qualify-mask15.py
python tools/prod_size_qualify_test.py -v
```

Freeze manifest: `evidence/2026-11-01-production-startup-qualifier/manifest.json`.
Root retains dispatch/receipt ownership. Do not submit the old unqualified
ordinary-launch file or reinterpret packing fit as a measured performance win.
