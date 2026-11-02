# PR graph-owner: helper-only capture-mode scope

November 2, 2026. Narrow final integration allocated from new research be8745778ebe145e66db2d1dea0194045ccad661, docs/steering/2026-11-02-research-pr-cleanup-api-contract.md. Work began 14:28:31 UTC. Original event amendment fed4ad7fa3f22d882525c094246c05020bd22e55 and manifest correction 9744719 remain immutable, as do prior context-sync STOPs. Captain owns build/benefit and shared ledgers.

## Exact implementation and outcome

ONE append-only transformer tools/g_pr_graph_owner_relaxed.py derives the frozen owner-event source, changing only the retirement-helper mode scope, its error restoration, popped-context identity check and extension identity. New test file tools/test_g_pr_graph_owner_relaxed.py inherits all actual prepare/Lt-run/native identity gates. No second performance candidate or native/device/math/fence/ordered-flag change.

Candidate: SHA256 4c6d3f13e0b9c72c4d4ab67f440a35f6ae39a954c818f78d35f3ff3d9090423b, 881883 physical LF bytes, final newline and original notices retained. Exact reverse through frozen event transform restores ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. All seven selected native factory sources/ordered flags/routes remain equal. Generated host/candidate/generation pins and CPU transcript are under evidence/2026-11-02-worker-pr-graph-owner-relaxed/; publication-manifest.json identifies exact published blobs and the enclosing commit.

Decision: SOURCE_INTEGRATION_CPU_PASS_TARGET_HOLD. 21 tests PASS in 35.405 seconds. No host syntax/build/link, compiler, install, Driver/GPU call, challenge CLI/service/check/public/default action. Final private check remains RESERVED. No runtime, numerical, useful compression, capture eligibility or speed qualification is claimed.

## Scope and exact failure handling

Only the dedicated retirement helper, AFTER the no-CUDA user callback signal is observed, initializes a mode variable to RELAXED and calls cuThreadExchangeStreamCaptureMode. The returned prior mode is retained, not guessed GLOBAL. Only after successful entry does it invoke mapping cleanup with that prior-mode pointer. Normal completion exchanges back to precisely that prior mode before deleting the already-cleaned mapping. Neither allocation/setup nor caller Lt/prepare paths change capture modes; callback remains lock-free signal-only, nonblocking and no CUDA. No global flag or PyTorch hook changes.

Every cleanup CUDA result is checked. On a cleanup/context push/pop/identity failure, the helper restores the saved mode BEFORE the existing fatal abort; no failed unmap/release is retried or presumed successful. If entering the scope fails, no cleanup or speculative restore runs. If normal/error restoration itself fails, the helper aborts without continuing; no retry or hardcoded-mode fallback. Error restoration does not pretend partial resource cleanup succeeded. A popped-context identity mismatch also restores then aborts. This is a no-throw/failclosed policy, NOT recoverable cleanup and not native-tested. Module nodelete and retained context lifetime are unchanged.

The previous event provenance remains mandatory: events are fresh, noncaptured, owner-specific, recorded after every actual eager Lt/copy enqueue; captures have graph refs and no event waits. RELAXED is not used to legalize encompassing context synchronization, graph-captured event waits, shared-resource destruction or context reset. The generated HOST still contains no cuCtxSynchronize/cuStreamSynchronize.

## Primary-context ownership boundary

The mapping releases exactly the lease it actually retained, after all its host/tensor/graph readers and eager events retire, and after its context stack is popped. Other legitimate shared-primary-context users must independently retain their own usage (including a live runtime lease where applicable); with their reference outstanding, releasing this one balanced lease cannot be the last destructive reset. An active capture ALONE proves neither that independent retain nor the total refcount. No count is queried or inferred from capture. External premature release/reset/shutdown remains outside this contract; RELAXED does not repair it. Target runtime ownership remains untested.

## Validation, gaps and return of ownership

The 21-test suite includes actual final prepare, before-plan integration, both PR Lt-run paths, independent state/stream ownership, deepcopy/fallback, clone/abort/inflight lifetime and post-enqueue event failure gates from the frozen event amendment. Added source gates locate the exact helper entry/cleanup/normal restore sequence, preserve the prior-mode value on cleanup-error restore, ensure callback/caller sources have no exchange, and forbid encompassing waits. Added CPU failure matrix covers three prior modes and normal/entry/cleanup/context/primary-release/restore failure exits. These helper-mode cases are explicitly synthetic source/model evidence, not execution of the C++ or CUDA API. In particular, mode-exchange/cleanup errors and context ownership are not validated on a target.

No primary survey or additional researcher was launched: this integration uses the published be874 contract and its primary-pins.json. No blanket capture-independence follows from mocks. Exact source ownership returns to captain for its separately owned offline build-gate and end-to-end benefit decision. No automatic dispatch, repair variant, proof loop or readiness cycle. A concrete mode/event/refcount/context contradiction stops this exact source. README/shared logs remain untouched.

