# Captain admission check: fixed G2/24 shared-memory fit fails

November 1 Pacific / November 2 UTC2026. Evidence base206ccfbc82cb15066f12e66b21cccee2cf82308b, original expert-pipeline-next report and source sizing script8b89f01e8dab6824a3e02eae112ea016eb6bc69411d98f80ac135228d4f23cd7. This is a new source/accounting result, not a GPU failure, native occupancy query or performance result.

## Decision

STOP the planned115968-byte-per-CTA fixed layout under the documented1024-byte driver reservation. The earlier fit calculation compared only user shared storage to SM capacity and omitted per-block system storage. I should have checked this before funding the broader implementation. The independent check caught it during the first implementation interval, before any compiler or GPU quota was spent. No full rewrite, header/ring/shape rescue, target job, public/scored/default change follows automatically. This closes the exact assumed layout, NOT all two-CTA expert designs.

Worker paused expensive implementation after captain's early warning and independently confirmed the header rule. Its initial append-only tools/g_expert_g224.py is unfinished WIP, not a runnable/qualified candidate. Intended launch bounds320,2 are uncompiled. Captain sent the verified STOP by message9e79185d-17b3-4ca9-bd6f-19b2e3a2b09f to existing worker ordinal7; worker owns a bounded unfinished-artifact/counterexample publication. The35/90minute full-build targets are superseded by this early-stop gate. No new thread/subagent/workflow was used.

## Primary-source check

NVIDIA CUDA13.0 Programming Guide, compute-capability9 shared-memory section, states the1KB system reservation. The runtime device structure exposes reservedSharedMemPerBlock as driver-reserved bytes per block. Installed CUDA13 occupancy header163b8f296ef2a33c799e2dc8c525b37166da9217bcc6a3c1d26b1c10aca5e22b adds that reservation to function static storage, then adds requested dynamic storage and rounds to allocation granularity. Its SM90 branch uses128-byte granularity. In that header the relevant arithmetic is at1414 onward; the granularity function starts620. A host C/C++ compiler was not found in the inspected WSL/default Windows PATH; we did NOT execute the native occupancy calculator or install a compiler.

Primary references retrieved November2UTC2026: [CUDA13 runtime execution/cooperative launch](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__EXECUTION.html), [CUDA13 Hopper guide](https://docs.nvidia.com/cuda/archive/13.0.0/hopper-tuning-guide/index.html), [CUDA13 Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#shared-memory-9-x), [device property reservedSharedMemPerBlock](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html). These are primary documentation, not an actual H200 property receipt. Retrieved guide values supply architecture assumptions; actual target property remains unqueried.

## Reproducible necessary-condition witness

[accounting.py](evidence/2026-11-01-captain-g2-admission/accounting.py) executes only the exact layout function AST from the original pinned design script, not its network discovery/main body. It reproduces the original22240/22272-byte baseline header and proposed23776/23808-byte header/ring start, then computes the same115968-byte request. It checks the installed header's source equations and records its hash. WSL existing CPU Python exits0; source-binding assertions PASS. [accounting.json](evidence/2026-11-01-captain-g2-admission/accounting.json) retains the exact result.

With zero additional static bytes and reserved1024, allocated storage per CTA is roundUp128(115968+1024)=116992. Two CTAs require233984 bytes,512 more than the233472-byte architectural capacity assumed in the design. Thus shared storage alone permits at most one such CTA under these assumptions; adding registers/progress proofs cannot fix this particular deficit. With reserved0 the old accounting fits; reserved768 fits exactly. This sensitivity is retained to distinguish an assumption from a measured target property.

To satisfy just this necessary shared-memory inequality at reserved1024/static0 requires dynamic<=115712, at least256 fewer bytes per CTA. Additional static storage and other allocation constraints can reduce that ceiling further. This is a reopening condition, NOT permission to shrink rings, silently alias memory or run a parameter sweep. No source-derived smaller exact header/live-range proof is provided by this turn.

## Independent launch/compiler finding

The pinned public source contains selected MAIN kernel __launch_bounds__(NTL,1), and its host path uses a checked occupancy call plus cooperative launch. ShrinkingNTL cannot inherit the old96-register result automatically: launch bounds influence compiler register limits. The fixed two-resident-CTA intention must be represented and verified in emitted code, then checked by the actual target occupancy API for the selected boolean, actual block size, static and dynamic shared memory. CUDA's cooperative grid limit depends on that occupancy times SM count; merely removing grid<=SM from the host guard would not prove admission. No new emitted resource row exists.

## Priorities and gaps

Zero new performance-qualified winners; top100 remains unachieved. This early source STOP is useful because it prevents a predictably invalid target launch, not because a report advances leaderboard performance. Current worker task is only publishing its scoped stop artifact. Scalar remains offline/unqualified and no automatic GPU allocation follows. Public incumbent and remaining quota are unchanged by this task; no fresh rank/quota query was made.

A future two-CTA proposal needs either authoritative target reservation that changes this arithmetic or a new complete storage/lifetime proof reducing actual allocation enough while preserving the full pipeline. An unused-field removal may be a real source opportunity, but its sizes/live ranges must be demonstrated and separately selected; a hypothetical256-byte saving is not an implementation. Compare that engineering cost with another complete base or the existing scalar treatment before reallocation. Do not re-open seam/RMW, substitute a smaller design merely to pass, or count unfinished generator WIP as a candidate.

Validation performed here: exact source-linked CPU layout/accounting, installed-header inspection and primary-documentation reconciliation. Not performed: native occupancy execution, device property query, CUDA compile, GPU qualification, correctness/timing tests or submission.
