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
