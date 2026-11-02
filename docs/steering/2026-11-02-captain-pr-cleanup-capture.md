# PR graph-owner review: context-wide cleanup is not capture-independent

November 2, 2026. Prior goal turn3102263 made progress: accepted independent source prooffcf5452 and isolated a concrete header-path risk. This turn revalidated the exact live worker ordinal28/queued29, reviewed actual generated host/entry code, independently reran13CPU tests, inspected installed Driver header signatures and found a new correctness blocker before GPU use. Top100 is not achieved; no new performance-qualified winner/public/default change.

## Measured/source evidence, not an observed GPU failure

The inspected mutable tools/g_pr_graph_owner.py has SHA3c1b45537d9cc8d8c237d93ee2d4a0a14b1c9c3d6d27e54e983535ab1ff996ce and HOST literal SHAedf8da0afd94bfc16cf107ff7ed3045a475abf5041aecc75e5e5a60e08193bd4. These are the preliminary revision at inspection, NOT a final worker publication or upload pin. Its cleanup helper calls cuCtxSynchronize; no cuEventRecord exists in that HOST. The callback itself only writes an atomic flag. Independent actual-entry/source CPU rerun: **13/13 PASS in14.085s**. These tests model refcounts/fallback/deepcopy/capture hooks; they do not establish safety of a real context-wide synchronization during an unrelated capture. CPU PASS does not override this gap.

[Source-counterexample script](evidence/2026-11-02-captain-pr-cleanup-capture/source-counterexample.py), [captured output](evidence/2026-11-02-captain-pr-cleanup-capture/source-counterexample.json), [CPU transcript](evidence/2026-11-02-captain-pr-cleanup-capture/captain-cpu.txt). The script uses a SYNTHETIC documented-contract schedule, not a CUDA implementation or target reproduction. It does not assert that the real judge has exercised this schedule.

## Specific contradiction in the current cleanup design

CUDA13 Programming Guide6.2.8.7.3.2 prohibits device/context synchronization while an encompassed stream is capturing. Moving cuCtxSynchronize to a helper solves the destructor callback's no-CUDA restriction, but does not limit the synchronization to that owner's resources.

A valid ownership/capture schedule remains possible: independently prepared ownerA has an eager Lt read, A's host owner retires, its helper is delayed, unrelated ownerB starts capture in the same context, then A's helper resumes and requests context synchronization. A has no future uses, but B's active capture is encompassed. That request is prohibited; the helper can invalidate/interfere with another capture or hit its fatal cleanup path. This is a source-contract counterexample, NOT observed judge corruption, a measured runtime error, or a general rejection of graph user objects/compression.

The ownership proof in researchfcf5452 remains useful: graph resource references can cover captured readers independently of Python. The newly invalid clause is context-global eager-retirement completion under independent capture, not those reference semantics. Source needs a completion mechanism scoped to the actual owner's noncaptured uses, not a weaker lifetime gate or a private evaluator hook.

## One narrow repair-or-STOP, no target allocation

Existing queued ordinal29 was EDITED, not promoted to Steer and not duplicated into a new thread. Original ordinal28 keeps its30active-minute implementation/publication budget. On that queued review, the same worker may use ONE <=10active-minute narrow cleanup repair plus actual-entry CPU tests OR report a concrete STOP. No generic rewrite, second performance variant, toolchain install, CUDA compilation, driver/GPU/CLI/check/public/default action is authorized. Final private check remains RESERVED.

Candidate discriminator: record owner-specific completion events AFTER actual eager Lt submission, outside capture; copy completion must also remain covered. Graph references already protect captured uses. The helper may wait only the owner's noncaptured completion records, never the encompassing context. This proposed event protocol is NOT implemented or qualified by this report; recording, retention, failure, event destruction, concurrent callbacks and exact reader coverage need proof/tests. Do not merely remove synchronization or treat a captured event as an eager-completion record. Recheck other cleanup calls against capture/context contracts too; resolving this one synchronization contradiction does not certify the rest.

Require actual generated _graft_lt_run before/after hooks, both PR plans, independent states/streams, deepcopy, no-future-read retirement, event failure/failclosed behavior and a synthetic A-retirement/B-capture negative that distinguishes the old protocol. Keep numerical/native/fence/flag/ordinary-fallback invariants and callback nonblocking/no-CUDA/DSO/context lifetime. A no-treatment fallback is not a compression winner. Captain reviews the published actual code before any later target proposal; no automatic dispatch after CPU PASS.

## Build gate narrowed with real installed evidence

The current WIP now explicitly supplies CUDA_HOME/include with with_cuda=False; initial missing-header-path risk is addressed in source, not proven compiled. Installed secondary CUDA13 cuda.h is1,280,649B/SHA7a79d67ca888061b1dfbe4739b51ad64f85a2f3ad874947ce09b2c635b4c7072. Its default cuStreamGetCaptureInfo macro selects_v3; the seven-argument declaration includes edgeData, matching the inspected call. User-object/retain/property signatures are retained in [installed-header evidence](evidence/2026-11-02-captain-pr-cleanup-capture/installed-header.json). Legacy declarations also exist; they do not change the default macro.

This is NOT target-header, host-build, link or Driver-support proof. WSL PATH lookup found no g++/clang++/c++/cc; this is only that bounded lookup, not a systemwide/compiler-availability claim or permission to install one. No compiler was invoked and no driver library loaded. Host C++17 requirements inspected in the installed Torch header are compatible in principle; actual target ABI remains untested.

## Priority and stopping criteria

Finish the one bounded source cleanup question; stop the exact context-sync protocol from target dispatch now. If a narrow owner-scoped completion proof cannot be delivered within the queued bound, archive a scoped source STOP without another readiness loop. If it survives, independently inspect implementation/host-build/eligibility/useful-compression uncertainty and expected weighted benefit BEFORE deciding whether the remaining private check is worth spending. PR affects only two long loads; raw bytes/API presence are not exposed-time or speed evidence. Fresh incumbent profiling is already completed; serial remains PARK. No repeated control check/unchanged competitor inventory is justified by this review.

Primary section: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#prohibited-and-unhandled-operations ; independently pinned full page bytes/SHA are already in41b3d80/fcf5452. No public score/rank refresh or new result is claimed.
