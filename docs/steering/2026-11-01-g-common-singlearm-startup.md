# Startup-gated single-arm common/RMW source

November1,2026. Bounded source/CPU work requested while the campaign is paused.
No GPU execution, official/service API call, dispatch, queue change or production
replacement. New AGENTS/CLAUDE principles at root `fcc15d5` were read before
continuing work. README/integration remain root-owned.

**Result:** one reusable single-arm startup qualifier fits below1MB and passes
seven actual TorchCPU orchestration tests plus scoped C++/real-CUDA-header gates.
It can emit one fixed short-RMW treatment or its common control; these are
**one implemented candidate plus a control**, not two independent candidates.
Neither source is H200-qualified. An authorized later scored attempt can run
the fail-closed startup gates itself; private comparison quota is not a source
prerequisite. Human resume, service/prior-job reconciliation and root's decision
remain prerequisites to any dispatch.

## Exact pipeline and reproduction

`tools/g_common_release.py::build(root,arm)` accepts only the two known fixed
inputs by rebuilding from exact607 and checking their complete source hashes:

|arm|strict prepared-source input|startup-gated output|bytes|
|---|---|---|---:|
|common control|`4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280`|`4e3895be33d3d8f60670b4e8138eb09654ddc6d64aa930170dc89cc3f019d270`|914084|
|fixed short-RMW|`2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd`|`97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7`|914572|

The closed native-gamma treatment is excluded. No arbitrary-source pin override
or runtime timing selection is supplied. Root's disjoint common/RMW composition
is reused, not reimplemented. Both retain attribution/license/NOTICE text.
Generate with `python tools/g_common_release.py ROOT OUTPUT --arm short-rmw`
(omit the option for the control). Scratch outputs are under
`.build/today-moe/common-release`; manifests are tracked alongside this report.

Every existing `kernel_fn` AST is identical to its fixed input. Every native GPU
function, Params layout, dispatch choice and launch argument remains unchanged.
Native resource exports append host-only code. ATT7's prepare-only host audit
has an exact reverse-source proof; its device prefix is byte-identical. Existing
preparation/tuning/startup checks are retained. Newly added preparation contains
no Events, profiling, comparative benchmarks, score model or path picker.
Shared reference constants are checked AST-identical to existing product globals.

This is a new host/qualification source epoch, not byte-identical4cbb or2884.
It does not prove the whole Torch/NVCC extension compiles, actual GPU capture
works, preparation fits295s or the new arm is faster.

## Actual resources and successful native selection

The inherited launch fix needs observed runtime policy, not guessed Python
plans. A thread-local ATT7 witness is disabled by default. Explicit prepare-only
begin/end exports enable it only while the private model is called. All three
actual launch paths record **after checked launch API success**, after the
original resolver and occupancy fallback selected KernFn/q.coop/flat/grid:
cooperative, extended cluster or ordinary. Final synchronization catches async
execution errors. Kernel pointer tags and27metadata fields report the selected
instance, actual q fields, shared/thread/function attributes and admission.

The ordinary launch is unchanged with a prepare-only cudaGetLastError check;
the extended path keeps its original disabled-audit branch exactly. When audit
is disabled, its hook makes no CUDA queries or record writes. End forcibly
disables and clears the witness in finally before any production state returns.
No observer executes on graph replay or chooses inference work.

Python counts successful `_decode_attention7` returns **after** the native call.
Both original normal/graft paths may catch a RuntimeError and use FA3. Such an
output can look numerically correct, so qualification requires one successful
native attention call, one actual27field launch record and one expert selection
per captured model call. Window, geometry, boolean domains, actual waiting/
cluster launch kind and admission are cross-checked. Legitimate unsupported
normal/fuzz fallback paths remain unchanged once the temporary observers are
removed; a noncanonical weight interface returns with an explicit UNEXERCISED
marker, not a qualified-DONE claim.

Expert getters inspect both actual false/true specializations for all five native
module keys. Two10field vectors record registers, local/static/dynamic bytes,
threads/maxthreads, blocks/SM, SM count, cooperative capability and capacity.
Existing G main/B4 local memory is **recorded, not forced to zero** using an
unrelated owned-kernel rule. Existing cooperative launches still check actual
admission on each call. Tiny/Tiny2 carry the RMW define only in the fixed RMW
arm; MID/main/B4 cannot. A closed gamma or stage-disable flag is rejected.
Marlin is the only existing native expert fallback and retains its own format.
Its adapter is a cyclic owner not supported by the generic cloner. If normal
prepare selected that fallback (format, build or resource reason), the qualifier
returns the original production state **before cloning**, with an explicit
UNEXERCISED marker, `rmw_exercised=0` and the existing backend status. No72check
DONE stamp is printed. G56 and ordinary valid-input fallback behavior remain.
A valid fallback result is not an RMW measurement or promotion.

Runtime identity captures actual Torch/build-CUDA, driver/runtime/NVCC versions,
GPU name/UUID/capability/SM count where reported. Missing/compiler-host-adapter
fields remain null; HOSTNAME environment is labeled an unconfirmed physical
node, not a hardcoded old runner identity. No network lookup is performed.

## Private full-model checks

One normal preparation is followed by **one private alias-preserving state
clone**, reusing the audited whole-G ownership factory. Weights/packed tables
are explicitly read-only; mutable tensors, nested Lt workspace and all storage
aliases are cloned. Unknown owners and production/private writable overlap fail.
The new pre-expert effective coefficient is copied, not silently shared. There
is no second opaque prepare closure or extra comparison arm.

All12scored model cells use the existing official-style natural input generator:
token loads1/8/16/32/128/256, sliding/full, page1/context80000, one fixed base
and independent fresh bank per cell. The busiest local EP range is selected from
the reference's unchanged router; inputs are not routed to manufacture a win.
The FP32 and FP8-reference adaptive math, routing and new-cache checks are reused
unchanged, with finite-value/norm/adaptive guards. Source changes preserve all
BF16 boundaries independently of numerical tolerance.

Each cell performs two observed eager calls, one observed initial graph replay,
and three individually observed fresh/base/fresh replays: **72full-model checks,
12model captures and36changed/restored replay checks**. The output and actual
new KV slots are poisoned; inputs are restored after output poison to preserve
legitimate input/output aliases. Every graph replay is followed by synchronization
and reference validation, not just the final replay of a long queue.

Selected expert empty captures poison output **inside** the graph, use valid
all-nonlocal local IDs and run on the same private scratch without counter resets.
They interleave with occupied model calls: **60empty checks,12empty captures,
48observed occupied restorations**. Existing production expert56startup checks
remain. All cloned guard sidecars and newly registered attention guards are
checked; private guard-table entries/IDs are removed afterward.

Temporary Python bindings, TLS mode, TF32 flag, dequant cache and guard metadata
are restored in finally, including failures. Private input/cache/graph references
are released after synchronization; the original production state is returned.
The complete bounded JSON report (<11000characters) is printed only after these
gates/cleanup. The clock starts before normal prepare and fails at295s around
each cell/replay and before return. It cannot interrupt a blocking compiler/CUDA
call, so setup-time success is unknown until actual execution.

## Local validation, fixes and remaining gaps

`test_g_common_release_host.py` executes the generated actual preparation,
owners, fixed selector, successful-return witness, empty/capture orchestration,
real routing/cache/finite/adaptive oracle and cleanup under TorchCPU. Native
arithmetic/CUDA execution and attention planning are explicit small CPU stand-ins.
Both fixed source choices pass; all72actual oracle invocations occur per arm.
The tests reject caught native failures that otherwise return correct FA3-like
outputs, wrong ordinary/cooperative metadata/window, stale empty/model capture,
nonfinite reference/output, bad cache, shared workspace/scratch, guard overwrite,
wrong arm flags, failed residency, inner deadline and fatal CUDA error. Six tests
pass in9.980s on the final bytes; this is not a model/kernel math result on a GPU.

The first native getter host-header check rejected implicit function-pointer to
void-pointer conversions. Explicit `(const void*)kernel` casts fixed that ABI
issue. Peer `1142844` independently compiles the exact getter and all new ATT7
host code against real CUDA12.8 headers, with disclosed minimal Params/Cfg
adapters and one replacement for the **unchanged** CUDA chevron launch when
using a CPU compiler. The native C++ mock preserves nonzero local bytes and
rejects all five injected CUDA API errors. The ATT7 mock proves disabled zero
queries/records, cooperative/cluster records and nine API/schema failures,
including driver/runtime identity errors. This does not link a full Torch
extension, execute a CUDA API or establish GPU capture/admission.

An initial stale-graph negative fixture left its mock graph-kind unset; its
accounting TypeError was fixed, after which the real poison/stale gate rejects
as intended. No candidate source defect was hidden by that fixture correction.
The first PowerShell compiler invocation mispassed include arguments; a Python
argv invocation supplied the exact intended header paths. Final real-header and
mock returncodes are recorded; failed command plumbing was not a kernel failure.

Focused resource/header peer scope is clear at the exact two output pins.
Final full-protocol peer seal is pending on the corrected pins at this owning
commit and will be integrated separately by root. No H200-ready or whole-model-success claim is made.
All actual H200 setup/capture/resources/correctness and official13/12 outcomes
remain missing. A printed source manifest or CPU pass cannot supply those.

## Decision

This is a **startup-gated, single fixed-arm measurement source**, not a private
comparator published as a submission. After human resume, service/accepted-job
reconciliation, final independent review and root's authorization, it offers a
way to use an expiring scored slot without requiring a private paired win first.
Its own startup must pass before the normal official correctness/timing phases.
No action is authorized while paused or API530/own prior result is unresolved.

Report the owning terminal job, all startup counts/resources,13tests/12benches,
raw+adjusted times, score/CI and hardware/evaluator gaps. A single-arm score
cannot isolate RMW's causal contribution from common repairs or fresh-run noise;
private matched attribution remains a separate question. Stop on any source/
proof/owner/native-coverage/finite/cache/capture/guard/resource/budget failure,
or a valid measured loss. No new candidate, mask sweep, gamma retry or automatic
submission follows this source handoff.

## Late fallback correction, before qualification

The first owning source `15c3a84`/bded4957+a12eb0d5 is **superseded**.
Independent protocol review found that valid nativeFalse states contain the
real `_ForkMarlinAdapter`, which stores its state (including its private bank)
cyclically. The generic whole-G cloner rejects that unknown owner before the
old downstream fallback resource branch could run. Six nativeTrue facade tests
did not cover that case; no GPU or scored attempt used those bytes.

The narrow fix detects the original Marlin fallback before private cloning and
preserves its adapter, bank, workspace and state identities. No opaque owner
handler is added. A seventh test extracts the **actual adapter class** from the
generated source, creates its real state cycle/bank/workspace, verifies the old
cloner still rejects it, and proves the normal prepare path returns untouched
with UNEXERCISED/no-DONE and all bindings/TF32/deq/TLS restored. All seven tests
pass on final4e3895be/97935e7d bytes. The C++/resource/header code is unchanged
from peer1142844; the fix is confined to the Python pre-clone eligibility branch
and its reporting. The unsupported-layout path remains explicitly unqualified,
not silently credited with the native72check coverage.
