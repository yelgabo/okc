# PR cleanup under unrelated GLOBAL capture — CUDA13 contract

November2,2026; bounded primary-only response to f3a93a2. No worker WIP was inspected. No implementation, compiler, GPU, Driver query, challenge CLI or quota use. Final check remains reserved.

## Concrete answer

**Noncaptured owner events do not by themselves clear the helper as-is.** CUDA13's documented helper-thread control is a scoped cuThreadExchangeStreamCaptureMode to RELAXED, with exact restoration. It removes the generic potentially-unsafe-API restriction imposed by another thread's GLOBAL capture, without changing that caller's capture mode or graph. It does NOT legalize intrinsically conflicting calls, captured-event waits, shared-resource destruction or last-reference primary-context reset. There is no per-function exhaustive unsafe list in the inspected primary docs, so claiming every listed function necessarily requires RELAXED would also overstate the evidence.

With immutable, successfully recorded owner-only noncaptured events, genuinely retired private mappings, and a valid non-destructive context lease, the listed cleanup has a **conditional source-level route**, not a newly identified intrinsic prohibition equivalent to cuCtxSynchronize. Actual source must implement/prove those conditions. Documentation does not guarantee particular internal synchronization behavior of cuMemUnmap, nor establish target-stack execution safety or contest eligibility. No blanket implementation clearance follows.

## Primary contract by operation

| Operation | Documented effect and capture boundary |
|---|---|
| cuEventSynchronize | Waits work captured by the event record. Here “captured” means the event's snapshot of previously submitted stream work, NOT necessarily CUDA graph capture. A recorded noncaptured owner event does not encompass later unrelated capture merely because that capture uses the same context. Waiting/querying a graph-captured event is intrinsically invalid even in RELAXED. |
| cuEventDestroy | Can precede completion; it does not block for event completion, and resource destruction is deferred. It is NOT a substitute for successfully observing owner completion before unmap. Its own return/error must be handled. |
| cuMemUnmap | Unmaps the entire mapped contiguous range; may free backing once mappings/handles are gone. CUDA13 explicitly notes synchronous behavior in most use cases. This is NOT a documented promise of no internal synchronization, but also NOT an explicit encompassing-context wait like cuCtxSynchronize. Free only exclusively retired mapping ranges. RELAXED removes generic unsafe checks, not a real dependency on B's captured work. |
| cuMemRelease | Releases one's valid generic handle; physical freeing waits for mappings and handle references. Releasing an invalid handle is undefined. Do not confuse handle refcount with GPU-read completion. No explicit context-wide wait or per-API capture exemption is stated. |
| cuMemAddressFree | Returns exactly the reserved address/size after unmapping. No explicit context-wide wait or per-API capture exemption is stated. Address reuse before all owners/readers retire is still invalid regardless of mode. |
| cuCtxPushCurrent/PopCurrent | Operate on the calling CPU thread's current-context stack. Preserve/restore that stack and verify popped identity; no changing another thread's context or destruction via cuCtxDestroy as a shortcut. Context validity and errors remain separate from capture-mode permission. |
| cuDevicePrimaryCtxRelease | Must balance an actual retain; it does NOT pop the thread's context stack. Last release automatically resets the primary context. That can destroy another user's context resources and is NOT made safe by RELAXED. A valid shared-context ownership contract must exclude last-release/reset while B still uses it; do not infer the count solely from an active capture. |

The event assumption needs actual provenance: same-context record after every real eager Lt/copy submission, never rerecorded into capture or destroyed/reused before the wait. An unrecorded event represents empty work and can report immediate completion. One record on one stream does not cover independent streams unless valid dependencies join them. Legacy/default-stream implicit capture conflicts during event RECORD are outside the helper's local mode; the original recording path must itself be legal. Merely labeling events “noncaptured” is not proof.

## Exact scoped-mode obligation, not a global weakening

Driver13 cuThreadExchangeStreamCaptureMode swaps the calling thread's mode with the supplied value and recommends push/pop use across module boundaries. GLOBAL is the default and can reject potentially unsafe calls when another thread captures globally. RELAXED removes that restriction on this helper alone; it still rejects operations necessarily conflicting with capture. THREAD_LOCAL also ignores other-thread capture checks, but this report uses the explicit RELAXED contract rather than inventing an exemption list.

The source proof requires a dedicated helper that is not itself capturing: enter once successfully, preserve the returned previous mode, perform only independent owner cleanup, restore that exact previous mode on every ordinary/error exit, and check restoration. Never use a hardcoded GLOBAL restore or toggle blindly after failed entry. If entry fails, do not proceed as if protection changed; if restoration/context push/pop fails, report the poisoned helper/partial-resource state rather than continue or silently retry a potentially already-consumed release. Mode exchange itself can surface prior asynchronous errors, so error returns are not proof the entire prior cleanup succeeded. Callback remains no-CUDA/nonblocking; this scope belongs only on the helper.

This does not synchronize B, relax B's capture error mode, insert graph nodes or modify timing/harness. Its justification is resource independence after graph references and eager completion, not permission to disregard real hazards.

## What remains unknown versus concrete STOP

- **Still STOP:** any cuCtxSynchronize encompassing B; any graph-captured event wait; last primary-context reset while another user remains; unmap before owner completion; CUDA calls inside the user-object callback. RELAXED rescues none of these.
- **Needs source proof:** exact event coverage, local-mode restoration, balanced context lease/stack and single-release failure handling. Without those, the event-only replacement is not cleared.
- **Primary-doc limit:** VMM API pages do not specify an exhaustive capture-safety classification or exact internal synchronization scope. Driver synchronization chapter says any API may block/synchronize for internal reasons and undocumented behavior must not be relied on. This prevents promising no interference/zero internal waits; it is not itself proof every API is intrinsically illegal during unrelated capture. No concrete blanket VMM prohibition was found in these sections.

Therefore distinguish a documented scoped-mode/independent-resource construction from target qualification. Captain retains allocation/eligibility/priority decisions. No last-check request or performance forecast.

Scope/pins: approximately14:15–14:21UTC, under6active minutes. Read current captain report/index; reused immutable CUDA13 Guide/Stream primary pages previously independently pinned in fcf5452, and fetched fresh Event/VMM/Context/PrimaryContext/synchronization pages. General web.run returned empty; direct HTTP succeeded. No immutable worker read was needed to answer the API question; no worker implementation verdict. Exact URLs/status/bytes/SHA and source assertions are in companion primary-pins.json.

Primary references: [Driver13 capture-mode exchange](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__STREAM.html); [Guide prohibited operations](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#prohibited-and-unhandled-operations); [events](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__EVENT.html); [VMM](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__VA.html); [context stack](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__CTX.html); [primary context](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html); [synchronization behavior](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/api-sync-behavior.html).
