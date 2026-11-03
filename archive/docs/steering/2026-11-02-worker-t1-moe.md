# Worker: held T1-only three-launch standalone MoE proposal

November 2, 2026. Source/host allocation `aa4e5cfec1f1b081ac50e3c62c53fdea511d282c`; original T1 implementation `0d26672d701d8ff15c4ba6d4681e1a545b05798e`, independent review `f17a7329`. Implementation, tests and evidence are in this report's publication commit. No new judge job exists. The valid-negative pre-gamma allocation/job `56e0c8b5634545e6` and its archive `db7800a3` remain closed and unchanged.

## Decision and exact deliverable

**SOURCE/HOST PASS, HELD for captain review; not target- or performance-qualified.** The requested minimal standalone adaptation is implemented. Recommend this exact frozen T1-only protocol for captain's consideration of a separately authorized finite measurement, not another source variant. No automatic check, repair, T8 expansion, compiler run or promotion follows this handoff. Existing public/default files and prior experiments are untouched.

- [Held payload](../../../docs/steering/evidence/2026-11-02-worker-t1-moe/payload.py): **370479 physical LF bytes**, SHA256 **`f2e11de2d0b30321437bbd55f32c8337886b762724eb914125dee7a439d87b96`**. Final newline present; no CR, compression or source encoding.
- [Manifest](../../../docs/steering/evidence/2026-11-02-worker-t1-moe/payload.manifest.json): `hold=true`, `dispatch_ready=false`, `gpu_qualified=false`, official arm `candidate2`, affected set exactly `[1]`.
- Runtime identity **`8d4c2ee6eb23aef4150eff34af387a65a6e5f85850432c9cb78139d93dd3118e`**.
- Strict reader `tools/t1_moe_read.py`, canonical LF SHA256 **`b1422b5cb05505b094c309c6fb4c11e5d9f7103de3a1e735bf9b632d3cca0ffa`**. This is a T1-specific reader; do not use the old pre-gamma affected-set reader.
- Exact ba9 base **`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`**; original control native **`c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`**.
- Frozen T1 native **`dbdd3381cc9b823afb9e7d8678bba536cd5d6e3731896153aee9908652237447`**, binding **`64986286b26616a9fe70c885b53aee9c409af0ed694cd16f1ece47df62193b75`**. Both match the frozen layer prototype's AST literals and complete ordered flags, including host `-std=c++20`.

## What changed, and what did not

New generator `tools/t1_moe_harness.py` extracts the original ba9 standalone native/packing dependency closure directly. Importing existing metadata/capture helpers does not build the pre-gamma package: a negative test makes that build raise and generation still succeeds. No pre-gamma native string, compiler descriptor, selector or runtime binding is emitted. Legacy `_expert_pair_arm` keys occur only in an explicit stale-state rejection check.

The same original five control build descriptors remain selected: tiny, tiny2, mid, B4 and MAIN. Exactly one additional descriptor is the frozen T1 three-launch module. Actual generated prepare/selection tests record **36 selected source/host/ordered-flag descriptors**: all nine sizes `1/4/8/16/32/33/128/256/257` in each of four placements. Both control placements use original T1; both candidates use three-launch only at T1. Both arms retain original control at every other size, including B4/MAIN. The manifest and host receipt retain the complete descriptors, not just family names.

The generator evaluates final original flag assignments before emitting constants; this matters because the original file has later redefinitions. Original `_fork_native_moe`, `_ma_ext`, layout/packing helpers and other extracted definitions are byte-for-byte reversible. The only edit within that original closure is replacing the legacy scratch guard allocator call with the existing standalone state-owned guard allocator. [Source validation](../../../docs/steering/evidence/2026-11-02-worker-t1-moe/source-validation.json) records every reversed definition's hash; the test suite independently repeats the reversal.

New `tools/t1_moe_body.py` binds `_t1_arm`/`_t1_arm_bound` only in private prepared state. Original control preparation and all five original module selections complete before installing the candidate module and scratch. `kernel_fn` uses the official candidate state; T1 reaches `t1_three_run`, while every other size reaches original `okc_run`. Crossed arm, stale pair metadata and unsupported/non-ep8 shard configurations reject rather than silently select another treatment.

No native source, helper, precision boundary, normalization, geometry, weight format, barrier, fence, stream or CUDA flag is edited. The frozen native retains BF16 gate/up rounding, FP32 down values/norm partials, complete per-expert norm and slot-ordered weighted BF16 combine. Native host binding still consumes `ma_w[10]`/`ma_w[11]` unit-packed weights, FP32 scales `[2]`/`[5]`, and BF16 gamma `[7]`. These are the original packing results, not pre-gamma or WGS-copy substitutes.

The native archived checkout has automatic CRLF expansion (physical checkout hash `5d031f4ac0bad4332f0b058472e60bde55b70ca372102e4a55df7236167f3257`). Its **repository LF bytes**, frozen candidate's native AST literal and emitted native all equal `dbdd3381...`. Source validation explicitly compares the repository blob, not physical Windows CRLF bytes. Originals were not rewritten.

The inherited extraction omitted the full trailing license block. This new payload preserves all original leading provenance lines plus the complete **14300-byte Apache/NOTICE/MIT block**, tested verbatim. This packaging correction does not modify the older adapter or negative archive.

## Scratch, ownership and cleanup

Each candidate owns three extra Torch scratch views with separate guard backings: BF16 `[8,4096]`, FP32 `[8,4096]`, FP32 `[8,256]`. Data total is **204800 bytes per candidate** plus `3*65536*4 = 786432` guard bytes, or **991232 bytes per candidate** excluding allocator overhead/transient zero tensors. Controls allocate none of these extra buffers. All original scratch/initialization remains intact.

The reused runtime inventories each backing storage across all four states, validates scratch/guard view ownership and rejects aliases/missing inventory. Deepcopy retains independent tensor storage; module metadata may be shared through the original immutable extension wrapper. Cleanup restores original scratch from the original scratch factory, zeros all extra T1 buffers, checks both guard lists, restores loader binding and resets graphs even on injected observation failures. No external allocator, graph ownership callback, process-global treatment selector or input/output cache is added.

## Measurement protocol and frozen reader

The validated protocol remains unchanged except state key names, additional owned guards/reset and explicit T1 affected metadata. [Runtime diff](../../../docs/steering/evidence/2026-11-02-worker-t1-moe/runtime.diff) enumerates those edits. The [contract diff](../../../docs/steering/evidence/2026-11-02-worker-t1-moe/t1_moe_contract.py.diff) changes only the affected-set declaration and its regression/drift predicates; the paired estimator, equal-six, duplicate-null and tolerance calculations are unchanged.

- Two independent controls/two independent candidates; candidate2 is official.
- Banks `1102031/1102037/1102049`, normal/skewed, all54 routing cases. Expected counts stay378 observed states,54 restorations,216 captures and1152 timed replays.
- Fresh/empty/distinct-occupied/restored captured-address observations, input restoration, output poisoning and guard checks are inherited unchanged. Same cold flush/event interval contains the complete graph, including all three T1 launches; no observer kernel or new measurement arm.
- Every bank/distribution/cross must pass equal-six factor `>=1.01` and log gain strictly greater than twice duplicate-null. Affected T1 regression `>=1%` rejects; all other loads' drift is reported, not hidden. No T8 expansion or pooled favorable banks.
- Same paired `.015` global/`.075` row tolerances and exact empty zeros. Actual official independent FP32 tests remain mandatory and **NOT RUN** here.
- Strict owning reader retains private/job/team/board/source/upload/log association and all5 official tests/6benchmarks/6profiles, integer seeds, complete counts, case order and source-mode requirements. The held reader cannot declare target qualification.

## Validation and limits

[Host test output](../../../docs/steering/evidence/2026-11-02-worker-t1-moe/host-tests.txt) and [host receipt](../../../docs/steering/evidence/2026-11-02-worker-t1-moe/host-receipt.json) contain the final actual-entry CPU result: **13/13 PASS in29.417s**, existing Torch2.10 CPU runtime. The suite executes the generated final `prepare`, all selections and `kernel_fn`, including inference mode, interleaved independent states, deepcopy, extra guards, failure cleanup, crossed/stale metadata and unsupported shard. Real reduced CPU packing checks distinguish w13/w2 values at w10/w11, independently execute original code/scale/unit layout operators, and verify global scales/gamma. Additional negatives reject native precision/packing/gamma/flag changes, wrong/missing compiler descriptors, stale/nonfinite/wrong output, scratch aliases/inventory, routes, deadline and reader identity/count/source corruption.

**These are real Torch CPU storage tests with explicit CUDA graph/event/native facades and reduced packing leaves.** Stub math is a simple synthetic function, not a MoE numerical oracle. Event values are synthetic all-one times, intentionally failing the performance gate; host timings are not device performance. The synthetic reader test borrows the old receipt's schema only and fabricates upload/log associations in memory; it is not a new judge result. The host-facade capsule is labeled by its filename and must not be used as target evidence.

Initial failed test transcripts remain archived: missing final flag evaluation was corrected before the final source freeze; two reduced-fixture expectation/fault-name errors were corrected without native changes. A later validation assertion initially compared LF native against the CRLF-expanded checkout; final evidence compares the unchanged repository LF blob. None of these are target/compiler observations.

No compiler, Driver query, GPU, challenge CLI, service call, package install, default/public action or subagent ran in this allocation. Historical frozen T1 secondary compilation is only prior evidence; this host adaptation has no fresh build/load proof. All target numerical, replay, memory/capability, four-state295s fit and performance gates remain open.

## Reproduction, next owner and stopping criteria

Existing WSL interpreter only, from repository root:

```text
/root/.cache/okc-host-validate/bin/python tools/t1_moe_test.py --evidence docs/steering/evidence/2026-11-02-worker-t1-moe
/root/.cache/okc-host-validate/bin/python tools/t1_moe_harness.py --out docs/steering/evidence/2026-11-02-worker-t1-moe/payload.py
```

Two in-memory rebuilds and the archived source/manifest compare exactly. `H_HOLD=True` rejects before preparation/build. No executable non-held proposal is archived.

Captain owns review and any new finite target allocation. A separately authorized measurement must still settle target C++20 build/load, independent FP32 numerics, complete replay/cleanup, actual295s fit and frozen performance gates. Previous272.22s preparation does not guarantee this six-build/four-state setup fits; fewer descriptors do not prove lower build cost. Extra two T1 launches, larger ordinary grids, dependent global intermediate traffic and lost overlap are real costs, not a speed forecast. No standalone result implies layer benefit or promotion. A concrete contract failure, failed fit/build/numerics, affected regression, valid subthreshold or ambiguous result closes this exact version without repair/tune/retry under this allocation. Stronger complete comparator evidence or a separately authorized measurement can change the roadmap, not today's source-only PASS.
