# Attributed layer fork: MoE safety and precision gate

Evidence pin aecd648 and this patch/audit commit. Root relays explicit user approval
for an attributed LAYER fork after ordering/precision/license audit. This report
does not authorize a public submission. No additional GPU check was dispatched.
The standalone external check b65c4c878014416c passes2.395703x, but it is a different
file and does not validate the repaired layer version or its ordering/precision.

Audited input: keell0renz8e30e1be994db4310794b8d04e90d18d895deb84,
`kernels/muse_decode_layer.py`, rawSHA256
1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a;
LF-normalized82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19.
Reproducible `tools/dl_fork_moe.py:patch_moe(source)->str` requires this exact
normalized input FIRST, before other fork edits. It keeps upstream notices.
Root must preserve the upstream Apache2.0 license/NOTICE and per-file author
notices; the repository NOTICE also includes other components' notices. The MoE
repair does not independently audit those other components or grant permission
beyond the user's attributed-fork authorization.

## Concrete findings and conservative selection

| path | input finding | repair or explicit fallback |
|---|---|---|
| all selected native epilogues | FP32 GEMM1 gate/up feed SiLU, skipping baseline BF16 cache1 | round both scaled gate/up to BF16 before SiLU; BF16 SwiGLU output preserved |
| all GEMM2 epilogues | FP32 GEMM2 result multiplied by gamma before rounding | round scaled GEMM2 output to BF16 first; then multiply FP32 |
| layer gamma | source forms FP32(1+rawBF16), baseline225 materializes effective BF16 | explicit BF16 effective gamma, then FP32 arithmetic |
| main WGS YBF | additional rounding of gamma product | disable YBF, retain FP32 product for norm/final sum |
| M16 helper0df | sentinel data flags use a separate publication protocol | replace alias by repaired helper0; no sentinel path can be selected |
| WGS xA/xB -> TMA | generic stores publish flags without global proxy bridge | per-writer global proxy fence before gather release, deferred whole-unit release, AND split-finisher release after its new stores |
| WGS weights slot | lane0 release after other lanes' generic shared reads | every reader fences and arrives; empty count NW becomes NT |
| WGS x slot | original retirement was lane0 per warp | retain original wait_group1/0 retirement points; every thread fences/arrives, matching NT empty count |
| skinny slot refill | TMA overwrite after CTA barrier without all-reader proxy bridge | every reader fences before the existing compute barrier/refill |
| MFIN | per-expert flag polling falls through after4M unsuccessful iterations; default removes grid barrier | force MFIN0; do not adopt this protocol on passing tests |
| final full barrier | relaxed RMW publication relied on implicit fence pattern | explicit acq_rel GPU RMW chain, existing acquire and CTA barriers retained |
| grid progress | ordinary launch lacked an enforced resident-grid guarantee | cooperative launch with checked SM90/cooperative support, occupancy, shared-memory attribute and grid bounds |
| prior PDL producer | cooperative grid must not depend on co-residency with an earlier primary | native MoE secondary PDL disabled, ordinary same-stream cooperative launch waits for prior work |
| generic non-PRMT main fallback | another unreviewed non-WS algorithm | never compile/select it; skip native entirely and use owned, or official Marlin when owned declines |

These repairs touch both selected CUDA source literals, main and helper0. The
helper0df alias points to the repaired helper0. The layer's M8 route is helper0,
NOT the standalone comparator's dedicated issuer. Source tests assert5 gate/up
and5 scalar GEMM2 round sites per literal plus all WGS/B4FIN output sites.
No reduction crosses the expert RMS norm, top-k weighting or BF16 output boundary.

## Ordering and progress reasoning

All generic readers/writers carry their own proxy fence before communicating
completion. CTA/warpgroup barriers then collect completion into the release
signal; consumers acquire before use. Split finishers need a fence AFTER their
epilogue, not only the flush-entry fence. Weight/x slot arrivals count every
consumer thread and keep the original ring phases. Async MMA retirement points
remain intact. The full global barrier's acq_rel RMW chain collects CTA writes;
its acquire plus CTA barrier precede final generic loads. These are applications
of [PTX8.7 sections8.8–8.9 and fence.proxy](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model),
not a claim that test success proves ordering.

Cooperative admission follows the [CUDA12.8 launch contract](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__EXECUTION.html):
grid<=occupancy*SMs and cooperative support are checked. This patch additionally
requires grid<=SMs,grid<=136 andgrid<160, matching fixed flag/fixup bounds, and
1<=M<=384 with local IDs0..31/-1. All CTAs can make progress, including a CTA
waiting for another's phaseA output. PhaseA has no dependency on phaseB.
The phaseB designated split finisher waits for later-CTA initial shares in the
ordered contiguous range; that dependency cannot cycle back to an earlier range.
All finalization waits occur after that CTA's own phase work. Per-launch scratch
is private and same-stream kernel completion protects reuse. MFIN's own
parity/count lifetime protocol is deliberately not certified or selected.

## Lossless scales and fallback reachability

SC6 preserves bytes in0x40..0x7F by storing byte-0x40 in6bits; PRMT eligibility
excludes invalid/unsupported byte values. Its existing pack functions are unchanged.
CPU exhaustive-value/mixed-bit fixtures verify the six-byte/eight-scale bijection.
Compression fallback remains plain8-bit scales when PRMT is valid but SC6 is not.
Packing uses state['ma_sc6']; native dispatch adds its SC6 define from the same
state. `_ma_ext` no longer consults the process-global compression flag. A two-state
CPU test flips the old global and verifies incompatible layouts cannot swap flags.

Outside PRMT or SM90, native repack/JIT is skipped. The proven owned adapter is
installed first. If its positive-normal-scale constraint declines, an explicit
official-Marlin adapter clones/packs a private bank and uses the baseline BF16
cache1/cache2/y interfaces and official fused post-expert finalize, with effective
BF16 gamma and already-local IDs. It warms outside capture. This adapter handles
all external callsites, including graft calls that bypass backend policy selection.
No host input-value synchronization occurs in the timed path.

All old unconditional native background/prewarm entry points become no-ops until
a prepared state exists. Original prepare's four legacy expert q/scale repacks
are removed only after AST proves each key has one store and no consumer. Every
backend-select branch rejects a forced unknown 'til' pick rather than reaching
the removed tensors. The post-prepare SC6 diagnostic cannot recompile native code
after a fallback. A200s preparation-age gate skips/stops additional native builds
and retains the installed fallback. A single long compiler invocation can still
exceed the judge's300s cap; no partial-check success is claimed in that case.

CUDA synchronizes outside optional catches after packing, before compilation and
before each eager resource probe, then after launch. Only a known prelaunch
FORK_MOE_RESOURCE rejection selects fallback; launch/synchronization faults are
fatal. Build failure selects an installed fallback and is printed explicitly.

## Validation, integration and expected impact

Four source tests plus seven actualTorch2.10CPU tests pass. They cover exact
source pins/selected site counts, scale-state isolation, legacy reachability,
duplicate fallback states, compilation/resource/budget fallback, fatal execution
errors, actual BF16 rounding, and the actual Marlin adapter body with CPU buffers
and op stubs. Stubs validate ABI/lifetimes, NOT SGLang GPU numerical correctness.
Logs: `muse_moe_w4a16/profiles/fork_moe_*cpu.2026-11-01.log`.
today_proj independently reviewed the earlier device repair and9-test host gate
(record8214527); subsequent changes are state/reachability/budget host fixes and
require the coordinator's final composition review. No local CUDA compiler exists.

Composition: apply patch_moe first; preserve/reappend HOST_BRIDGE AFTER projection
tail truncation, then append today_proj's `module_source(owned_source)`, NOT its
owned-only `patch_owned_moe`. Root already identified and owns this integration.
Keep code visible; do not strip C++ template lines. Check final AST/payload size,
all per-state flags and actual selected route status before GPU dispatch.

The unmodified external standalone screen improves published score8.12% across
runs, but repaired native throughput is UNKNOWN. Restoring barriers, fences and
BF16 boundaries may erase that margin. Historical raw median differences from
owned47 to external at1/8/16/32/128/256 are1.89/4.14/4.28/14.43/16.36/29.72us
across different inputs; these are comparison envelopes, not expected layer
fallback costs. Require official full-layer correctness and matched fresh/replay
gates, record whether native or fallback actually ran, then compare whole-layer
score. On any unsupported native route select owned/Marlin; on an execution fault
stop and diagnose. No public promotion based on the unmodified donor check.
