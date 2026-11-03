# Outgoing graph-query stop and bounded diagnosis

November 1 Pacific / November 2 UTC, 2026. Evidence base: worker terminal archive **26b3b88b7a98920d0717d6d8ac0af8660677b199**, independent monitor **b3f7e6c**, and selected query source at that revision. This decision supersedes the queued status, not the frozen experiment or its reader.

## Decision and measured outcome

Own private **9adfd4dbc98f4998** is terminal setup FAIL: passed=false, return112, no OUTGOING_REPORT, official tests, benchmarks, profiles or score. Frozen reader rejects; complete=false/private gate=null. This is an **instrument blocker, not a valid timing loss**. No retry, second GPU job, public submission or production change is allocated. Six private checks and five scored slots remained in the terminal receipt; this is an archived quota observation, not a new live query.

Canonical [worker terminal report](2026-11-01-worker-outgoing-pair-terminal.md) owns the raw receipts, log, manifest and precise timestamps. Accepted04:51:11.656UTC, started05:08:41.850UTC, finished05:11:58.703UTC. Queue1050.194s and worker196.853s do not establish prepare duration: prepare_s=null. Startup56 verified replays and WL_PREP four-arm common resource parity are retained. Exact secondary/census and cleanup-completion records are missing because the final capsule was never emitted. No complete target qualification is claimed.

## Independent source diagnosis: hypothesis, not localization

In tools/g_outgoing_seam_query.py, seam_graph_edges enumerates every kernel node in the mixed full-layer graph and calls cudaGraphKernelNodeGetParams. It does not restrict inspection to expert/unpool symbols. The exception identifies the binding but not its internal API, graph node, row or arm. A delayed asynchronous error is not excluded.

A concrete distinguishing hypothesis is runtime host-symbol lookup on a foreign/driver-created captured kernel. NVIDIA's CUDA13 Runtime Graph documentation lists cudaErrorInvalidDeviceFunction for this accessor. A primary NVIDIA forum user's captured-cuBLAS example reports the same accessor error; that is analogous evidence, not confirmation of this run's cause. Driver graph kernel parameters use CUfunction. NVIDIA's runtime/driver interoperability documentation provides cudaGetFuncBySymbol and the explicit cudaFunction_t/CUfunction cast bridge. A repair must put BOTH native symbols and graph nodes into the same canonical function-identity namespace; comparing driver handles against existing host-stub pointers, or silently skipping unknown nodes, is not acceptable.

Primary sources independently reopened November2UTC:
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__GRAPH.html (cudaGraphKernelNodeGetParams)
- https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__DRIVER.html (cudaGetFuncBySymbol and CUfunction interoperability; explicit13.0.2 document, not mislabelled13.0.0)
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html (driver graph parameter API; installed-header ABI must still be checked)
- https://forums.developer.nvidia.com/t/stream-capture-of-cublas-gemm/216148 (user's reproduction, not NVIDIA diagnosis of our job)

## Next actions, ownership and stopping gates

Worker retains sole dispatch ownership but has NO current GPU allocation. The existing delivered allocation is at most15minutes SOURCE/CPU after immutable terminal publication: a separately named scratch read-only host-query bridge, canonical driver identities for both symbols/nodes, named API/node/row/arm diagnostics and prefix resource/build receipts surviving failure. Check available driver headers/linking/ABI without installs. Preserve non-null edgeData, incoming/seam/final identity checks, device bodies, flags, timed code, fresh outputs and all existing gates. Use mixed-foreign-node and identity-namespace CPU/source negatives. Return a minimal delta and proof or a specific ABI blocker. Do not regenerate the full harness or begin another readiness cycle yet. Frozen f25d714e source and5435108e reader stay immutable.

Researcher continues the already allocated25minute SOURCE/CPU complete ruwwww@c035bf9 versus common4cbb comparison, due approximately05:27UTC: complete prep/dispatch/precision/progress/epilogue and own source/result/license, at most two concrete selected multi-load opportunities. No new agent or duplicate task is created.

Captain will compare that alternative with the bounded query repair before allocating any further measurement. A source/CPU bridge is not target runtime proof and does not itself earn another check. Reopen measurement only with a concrete diagnosis/identity proof, preserved full qualification, a complete fixed candidate and an explicit opportunity-cost decision. Stop repair if the ABI remains unresolved or scope expands. No generic knob sweep, observer-only GPU job, automatic retry or default change.

## Project accounting

ZERO newly performance-qualified winners from this lane. The failure neither supports PDLON performance nor closes outgoing PDL as a family. It consumed one allocated check without producing the intended performance decision; I own that planning/instrument risk. The next work must reduce a specific uncertainty or produce a materially stronger complete candidate, not merely add preparation artifacts. RMW promotion remains closed by its valid subthreshold paired result. Top100 remains unachieved.

