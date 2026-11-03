# Owned verify production pruning and dispatcher: independent checkpoint

2026-11-01. Root main `ec0bb05`; reviewer base `1ed36ee`; owned worker
base `4557de7` with uncommitted generator revisions reviewed by file hash.
No reviewer edits to root/worker files, GPU jobs or public action.

**Dispatcher host contracts cleared; owned native pruning awaits selected
compile evidence and a frozen final checkpoint.** Nothing here qualifies a
nonzero-mask hybrid for H200 execution, byte fit or public release.

## Owned backend source review

Reviewed worker `tools/dl_owned_verify_production.py`, raw file SHA256
`cccf37908c06656571105e82a0cf8692cc463c41e38baf3c1bf1dd8e9f24ea80`.
Input is the frozen private owned arm, pin
`384933697aadb2ebfc2abe672b216751e55fd5949e01c0c57d03caa586fe6fe3`.
Current both-token result is399155 bytes, SHA256
`6de124667ee5b56c81771b4fd5d2591dc62e1bcfbf60f18baa680438d543bf42`.
These are source-review checkpoints, not permanent clearance for later edits.

The first draft read its input from ignored `.build/` output. That prevented
clean-checkout reproducibility. The owner repaired `build(root)` to regenerate
through tracked `dl_verify_pair_slices.py`, assert the frozen input pin, and
restore the temporary import path. Changed native literals are now visible
raw multiline strings rather than escaped one-line representations.

Current source pruning preserves entire ONE/MOE active device strings.
`DL_ONE_INSTANCES` lists only `(8,3)` and `(8,51)`, and both native resource
query and launch switch consume that same macro. The cooperative-launch
probe uses separate generic counting kernels; it does not reference a removed
attention mode. Original `_one_setup` and its selected-plan/resource assertions
remain. Omitted fallback attention modes cannot silently take over: scoped
prepare asserts the same fixed3/51 plans as the frozen private arm.

The main extension retains route_fwd only, plus attn_boundary when256 is
included. The selected `_abnd` call does not use the removed occupancy export;
custom preparation does not call `_abnd_enable`. The retained main CUDA
prefix includes launch_ex, needed by the boundary's launch_cl in the joined
translation unit. The256 boundary restricts its geometry dispatch to the
original `<1,512>` geo4 implementation. The128 SG module constructor and
call remain; no custom128 boundary replaces it. WD128 retains its helper
prefix/tensor-map encoder and original dense-WD source; SK/GEMV registrations
that cannot be reached by the scoped Python entry are removed.

Independently executed the existing root contract tests
`test_owned_recorded_128_policy` and
`test_owned_three_lt_plans_and_closure_ownership` against the newly generated
owned backend, not the original private arm: both pass. The generated module
has no additional/unbound module globals. There are no retained decorated
Python definitions, so this draft closure's decorator omission is not a
realized defect in the pinned input (it would need a guard for generic reuse).
These are Python call/plan contracts, not native compilation or math tests.

Policy retained: QKV128 WD `b1n128c3s8h`; ordinary O128/256 GEMM;
unpool128 layout3 `[66,18,1,0,0,2,35,0,5]`; unpool256 layout1
`[66,20,1,0,0,4,35,0,3]`; QKV256 layout0
`[66,124,1,0,0,0,35,0,2]`. Lt closures retain private workspaces and
correct per-plan M/weight orientation. Main51 MoE ordinal10, effective
BF16 expert gamma, FP32 pool/router output, and final SG boundary remain
as in the already reviewed frozen private policy.

Remaining native gate: local selected-entry compilation/source equivalence
for route's active FP32/hi-lo path, boundary `<1,512>`, and dense WD128.
The compile adapter under review explicitly excludes host extension APIs;
device compilation cannot certify complete torch-extension host compilation.
No terminal compile results are attributed here before the owner supplies
them. Future deeper ONE/device pruning needs a new exact-source review.

## Root fixed-mask dispatcher review

Reviewed `tools/dl_verify_hybrid.py`, raw SHA256
`d004f6c6da0efed6030427805c154ba97a7644edfc0ba060e23394982c7a747e`,
and final body raw SHA256
`889d7e2cda9cafb3216b3808cd440a41ec050ddcbff383e78ed994ed8e0b17a5`.
Mask0 returns the byte-identical869076-byte607 incumbent. Nonzero masks
select individual verify16/32 sliding/full cells; other loads and unsupported
configurations/tensor metadata use G. The wrapper keeps separate backend
module namespaces and per-call states. It does not pick a mask from measured
timings or input tensor values.

One real production-domain issue was found and repaired: the owned private
prepare rejects unsupported expert scales, whereas full G may use its Marlin
fallback. Unconditionally preparing owned would fail the entire hybrid even
when G could run. Root now calls the exact owned `_w4_scales_ok` predicate
on both scale banks viewed as uint8 during prepare, before owned repacking.
Unsupported banks omit owned state, and kernel_fn delegates every cell to G.
There is no broad exception catch. Unexpected packing/compile errors remain
fatal. The final deadline guard now runs unconditionally before prepare
returns, including the unsupported-weight preflight path.

Independently reran all nine root Python host contract tests: PASS in0.273s.
They cover all16 masks/all12 cells, unsupported config and ABI metadata,
unsupported w13/w2 separately with no owned prepare, preflight expiry,
per-call state/module isolation, readable source transport and exact mask0.
Backend modules/events in these tests are host stand-ins. They do not test
real native resources, GPU capture, shared-library symbol behavior, complete
preparation or numerical outputs.

## Gates retained

No fitting nonzero-mask production artifact is established here. Final matrix
must be regenerated from frozen backends and current wrapper; earlier byte
budgets predate the fallback/deadline additions. Full G+owned runtime setup,
memory/resource admission, fresh-input correctness and completion within300s
remain unmeasured. A source-level dispatcher clearance does not certify a
future public hybrid. Root retains the exact incumbent and owns any next
qualification after reset. Reopen this review when the owner supplies frozen
selected compile evidence or changes a native body/policy, not by silently
extending this checkpoint to a later source hash.

## Final selected local compiler addendum

2026-11-01. **Selected local device-equivalence gate cleared** for the same
399155-byte owned backend, SHA6de124667ee5b56c81771b4fd5d2591dc62e1bcfbf60f18baa680438d543bf42.
Current generator raw SHA256 is
`916e8fb67ab8318bf8a2051895eeee95c71440cae1da369f18559ae36397d70e`;
compiler helper raw SHA256 is
`38178c725f7771b6e9f6724bebf467c696077a1ab58dce3f2e4b48322ccfeade`.
Worker evidence is in
`docs/steering/evidence/2026-11-01-owned-verify-production/` with six
original/production manifests, logs and selected-sass-comparison.json.

All six manifests pin original384933 versus production6de12466, successful
NVRTC/PTXAS status, compute90a/sm90a, C++17 and the same compiler options.
The local CUDA12.8 driver emits only selected expressions plus incidental
original globals; selected-resource comparison is per exact symbol, not an
aggregate over additional unused kernels:

| Selected entry | Instructions each | Registers | Barriers | Static shared | Stack/spill stores/spill loads |
|---|---:|---:|---:|---:|---|
| WD `<1,128,3,1,false,8,BF16,0,true,0>` |776|90|2|0B|0/0/0B|
| Boundary `<1,512>` |424|64|1|64B|0/0/0B|
| Route `<float,true>` |1664|80|1|20B|0/0/0B|

Independently read the raw SASS caches under
`/root/.cache/okc-owned-{orig,prod}-{wd,boundary,route}/` and compared the
selected opcode/operand lists. WD and boundary match directly after removing
address prefixes/outer whitespace. Route has only two differing operands:
RET instructions refer to each compilation's own anonymous-namespace kernel
symbol. Replacing exactly that selected self-symbol with a common marker
makes all1664 instructions equal. No instruction, register, constant,
predicate, memory operand or wait/barrier is otherwise normalized away.
Our independent hashes differ from the owner's formatting normalization;
both original/production lists match within this independent calculation.
Results are retained in
`evidence/2026-11-01-owned-production-review/independent-sass.json`.

The extractor adaptations were inspected: remove Torch/ATen/driver headers
and host extension launch sections; use CUDA runtime/cstdint headers;
replace the route's missing host-header INFINITY with exact positive-infinity
bits and supply the normal is_same_v primary/equal-type specialization;
declare the128-byte/64-aligned opaque CUtensorMap for device-only WD and
set existing SK_DEVICE_ONLY. SK_PDL=1 matches the selected owned flag. The
boundary extractor stops before host launch_cl, and the route extractor
replaces host launch helper bodies with declarations. These steps preserve
the selected device math but expressly omit host APIs, tensor-map encoding,
launch attributes and torch-extension linking. Thus this gate cannot certify
them. Earlier header-adapter failures remain archived and are not kernel
failures. ONE/MOE large active strings remain exact by source identity; this
pass does not claim newly compiled ONE/MOE entries.

The owner's three CPU tests were rerun independently with
`OKC_HYBRID_ROOT` pointing to the tracked root checkout: PASS3.460s.
They check generated globals/visible literals, exact selected function bodies
and projection/pin contracts. An initial run without that environment used
the older sibling checkout, which lacks the new tracked slice generator and
failed before tests; setting the documented root selects the actual current
tracked generator. This is a test-environment dependency, not a device
correctness result.

Root's final regenerated matrix reports **1,072,104 bytes** for a single
verify32 override and **1,125,548 bytes** for mask15; every nonzero variant
remains NOFIT. Those are root's final matrix figures, not a newly measured
GPU result. They supersede earlier pre-wrapper size estimates. Source/local
compiler review is complete for this checkpoint; full host linking, H200
execution, preparation under300s and a fitting production hybrid remain
unqualified. Preserve exact607 mask0. Do not submit or label a nonzero
hybrid shippable on the basis of these local proofs.
