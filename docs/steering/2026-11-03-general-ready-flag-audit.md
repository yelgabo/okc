# Independent Tiny ready-flag source audit

Actual pickup: 2026-11-03T08:42:22.919655+00:00, shared sourcefa862383. General takes the requested bounded read-only contributor/reset/publication cross-check for accepted MOE-SHORT-READY-01e3e2495464f34b2c, frozen payloadb6107a783c8ded6dc626f6cfa99e7e1b53201e5225bd2073f736c69125f16964, helper 974d7743bb8d049150fe46b62f84c2308158e382588903f185d8155c787faf20.

The deliverable is one completion or concrete counterexample on this selected source. The implementation owner retains its running job and sole collector 95504. This audit creates no GPU job, duplicate instrument, candidate edit or reply gate. Previous General jobs and the f401 target-export audit remain complete.

Pickup scope was actual selected source/flags, contributor intervals versus direct overlap, publication visibility across helper lanes/consumer warps, and reset storage, lifetime and progress against the retained grid protocol. CPU enumeration and source reasoning do not establish GPU performance or substitute for the owner's numerical/replay gates. No target outcome is assumed.

## Bounded decision

The contributor mapping and reset placement pass the source/CPU cross-check. Publication visibility for other helper lanes remains an explicit proof gap. This is a counterexample to treating the current prose argument as complete, not an observed GPU wrong output or a demonstrated PTX execution. General's requested review is complete with that limitation. No new optimization, GPU check, retry, reply gate, cancellation or source edit follows.

The [independent enumeration](evidence/2026-11-03-general-ready-audit/check.py) binds the archived accepted ID, exact paired payload and actual selected helper. It checks both candidate compiler requests select NW8/WS/Tiny with no SPLIT, and retains the original proxy fences, warp/CTA joins and final acq_rel counter sites. [Results and replay hashes](evidence/2026-11-03-general-ready-audit/manifest.json). These are source checks, not a target build review.

The frozen [paired source](evidence/2026-11-03-moe-short-ready/paired.py), [manifest](evidence/2026-11-03-moe-short-ready/paired.manifest.json) and [accepted registration](evidence/2026-11-03-moe-short-ready/accepted.json) bind this audit. Native line numbers below refer to the selected helper literal extracted from those bytes, not the outer Python file.

### Contributors and storage

For every grid 1–136 and skinny-job count 1–32, direct intersection of the 512 unit job with each proportional CTA range agrees with the selected first/last formula. Across 4,352 configurations and71,808 job intervals, all 358,814 positive contributions are covered exactly once. Every missing-contributor case leaves the original credit sum below 512. The 32 lane stride covers every required flag without duplication;2,297,856 lane assignments are checked. The largest tested intermediate numerator is2,244,607, below signed32 bit range. Nonzero njF offsets preserve the job-relative index. Omitting the last contributor is detected by the independent overlap oracle.

With positive jobs, every supported CTA owns at least one A unit because512jobs≥512>136. A helper therefore reaches q==nA-1 and publishes once. With no jobs, no B loop enters the division-based wait. The candidate does not use balanced or dynamic ranges. Each B compute warp waits before its first activation load for that job; its full-warp vote is followed by the retained warp join.

Actual scratch coordinates are ctr[0,8), old doneA[8,96), new flags[136,136+G), xready begins520 and alternate sum buffer begins1032. The maximum flag index is271. These ranges are disjoint, aligned int32 storage. The pointer offsets and zero-initialized allocator are present in the frozen paired file. This bounds the new footprint under the actual G≤136 host check; it does not validate arbitrary shared-scratch use across streams.

### Reset and progress

The flag reset is after the original helper join, compute join, final acq_rel grid-arrival counter, acquire wait for all G arrivals, and the following compute join. Thus no same-launch B consumer still needs a flag when reset begins, assuming the retained counter/barrier contract. Each flag has one CTA writer and one reset writer. All CTAs need not reset together; next same-stream kernel start follows completion of all prior-kernel threads. The source uses the current CUDA stream and an ordinary cooperative launch with a residency admission check. It does not newly enable overlapping launches using this scratch. See [CUDA task ordering in PTX8.7](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-synchronization).

Producer progress remains independent of B readiness. A helpers publish after their assigned A work; they do not wait for the B consumers whose flags they satisfy. The retained partial ring and B tail handoff remain unchanged. This is a dependency/source argument under cooperative residency and private ordered scratch, not a proof of GPU scheduling fairness or a new kernel-lifetime certification.

### Publication proof gap

Selected helper lines 1567–1570 are, in order, each lane's device fence, the helper warp join, and lane0's relaxed GPU flag store. Consumer lines 1409–1410 acquire the distributed flags and vote, then line 1440 performs the warp join before activation loads. For lane0's own prior stores, the fence and strong relaxed store form a release pattern. The other helper lanes' final epilogue stores are relayed to lane0 by the join, which occurs after lane0's fence.

The unresolved edge is a nonleader's final store preceding the first fence of lane0's release pattern. Consider nonleader store D1, fences F0/F1, the helper join B, leader flag store W, consumer acquire R and activation read D2. The source gives D1→F1→B→W and F0→B→W. It does not by itself show D1→F0. Choosing an SC-fence order F0 before F1 therefore defeats the simple argument that every helper lane's data precedes the first operation of lane0's F0/W release pattern. This identifies a missing proof edge; it is not a complete model-checked litmus proving flag1/data0 is allowed.

PTX8.7 defines a fence-before-strong-write release pattern, limits its direct effect to operations before the pattern's first instruction, and provides transitive causality through synchronization. Those clauses need an explicit argument for this cross-lane edge rather than the claim that a vote or relaxed store alone publishes the whole warp. [Release/acquire patterns](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#release-and-acquire-patterns), [causality](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#causality-order). This paragraph is an inference about the exact source, not an NVIDIA finding about this kernel.

CUDA specifies device-scoped ordering for __threadfence and participating-lane ordering for __syncwarp. Those guarantees explain the local relay, but the report has not established the stronger GPU-wide relay claimed after the pre-join leader fence. [Fence functions](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#memory-fence-functions), [warp synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions). A native CPU model with C++ release stores models a stronger publication operation than the selected relaxed PTX flag store, so its success does not settle this point.

The review does not conclude that the implementation is unsafe, that the existing job must stop, or that the control has a demonstrated defect. The current helper inherits part of the predecessor's fence/join publication shape, but ancestry and passing numerical tests cannot supply the missing argument. The implementation owner retains the ordering obligation before treating any passing timing result as release-qualified. No alternate candidate or repair sweep is allocated by this audit.

### Remaining evidence and ownership

The review read archived source and accepted registration only, without querying e3e2495464f34b2c or its collector. Actual GPU correctness, replay, failed-status preservation and complete-call timing remain with the original owner and frozen gates. Native publication/reset behavior was not executed by General. No second reviewer or subagent was used under the current no-agent instruction. The exact mapping/source checks above were executed independently; the publication concern is an unreviewed source-level reasoning finding and is explicitly limited to that claim.


Public boards refreshed2026-11-03 08:49:12UTC. Owned MoE2821 remains5737.988910us/2.3438x, with physical source correspondence unverified here; the other four owned best scores are unchanged. Required promotion guard in the Documents checkout passes with16numerical leads and zero disposition gaps. [Snapshot summary, full-snapshot hashes and guard](evidence/2026-11-03-general-ready-audit/manifest.json). This source audit creates no private performance receipt or promotion action.
