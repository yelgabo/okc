# Frozen Tiny readiness: contributor, reset and publication cross-check

November 3, 2026. Read-only research support for the existing MOE-SHORT-READY-01 endpoint, not a new task, implementation, approval gate or allocation. The root's separately authored reconciliation at fa862383 remains authoritative; the prior blocked draft/package was not applied. No target-export audit was repeated.

## Source identity and limits

Review pins: commit `73ab76cf508fc442ec7fd0910dad45280f0aa87c`, paired manifest `b6107a783c8ded6dc626f6cfa99e7e1b53201e5225bd2073f736c69125f16964` / 490593 bytes, selected helper `974d7743bb8d049150fe46b62f84c2308158e382588903f185d8155c787faf20`. The candidate helper was independently reconstructed from the archived, hash-verified `b8ed8543` helper and the four exact edits in `tools/moe_short_ready.py`; its complete 140185-byte SHA256 matches the frozen manifest. The full paired payload was read through the connector but was not independently byte-rehashed here. Native line numbers below refer to the reconstructed helper, not the enclosing Python file.

The accepted job `e3e2495464f34b2c` and collector95504 remain solely with their owner. No job-status request, compiler invocation, GPU execution, native/wrapper modification or target-result claim occurred. The existing General f401 export audit is already complete and is not this review. This publication adds only a review record; no owner queue or accepted source changes.

## Contributor and reset result: no counterexample found

An independent CPU enumeration checked all 71808 job intervals for 1..32 jobs and 1..136 CTAs. The native first/last formulas select exactly the CTAs intersecting each 512-unit job interval, and the lane-strided scan covers each contributor once. All 358814 single-withheld-contributor checks leave the original weighted sum below512. The largest formula numerator is2244607, safely within signed32-bit arithmetic. Zero jobs reach no B wait or division. Selected T1/T8 have no fat jobs or dynamic skinny pool; the original corrected `muldiv_g` partition matches the integer interval used by the wait.

Each publication is after the helper's entire A range, including ranges spanning jobs; no earlier-credit mechanism is introduced. The coarse dependency is retained, not eliminated. The helper's B-tail exclusion does not exclude the last A unit.

Flags occupy scratch int32 indices [136,136+G), below xready at520 and the alternate sum buffer at1032, and above doneA's88 words. The candidate resets them at native2451, after the unchanged all-CTA completion protocol at2436–2445. Every current B reader must have finished before this point; the finalizer does not read these flags. Same-stream completion and private-state ownership order the resets before replay, including empty calls and transitions to unchanged MID/WGS paths. This reasoning assumes the existing launch/residency and serialized mutable-owner contracts, not concurrent reuse of one state.

## Publication result: the written proof is not closed

The exact writer sequence at native1563–1570 is:

```cpp
store_intermediate_by_each_helper_lane();
__threadfence();
__syncwarp();
if (lane == 0) st_relaxed_gpu_flag(1);
```

The consumer retains lane-wise GPU acquire loads, a full-warp vote, and the original `__syncwarp()` before coherent phase-B `__ldcg` reads. I found no missing consumer join or non-coherent activation load in this selected path.

The unresolved point is the **nonleader writer-to-leader relay**. PTX8.7 section8.8 recognizes a fence followed by a strong store as a release pattern, but its coverage starts before the pattern's first instruction. Here lane0's fence precedes the warp join. The join orders other lanes' writes before lane0's subsequent operations; by itself it does not place them before that earlier lane0 fence. The owner's prose does not explicitly close this edge. This is an inference from the documented ordering rules, not an observed H200 stale read or a fully model-checked forbidden/allowed execution.

The existing CPU witness cannot settle that question: `tools/moe_short_ready_host.py` publishes with `std::memory_order_release` after its data writes, and models the vote using `std::barrier`. Those are deliberately stronger abstractions than the native relaxed-store relay and vote. Its missing-last-contributor negative is useful coverage evidence, not a native writer-publication proof. The original helper also used a fenced warp relay before leader atomics, so this review does not establish a candidate-only regression or condemn the baseline.

**One missing action:** close the nonleader-to-publication ordering argument against the frozen emitted instruction sequence and the exact PTX release-pattern rule. An explicit applicable proof or demonstrated stronger emitted synchronization would resolve this review concern. No payload edit, new experiment, cancellation, retry or waiting-for-research reply is requested. Correctness passes alone would not supply the missing formal edge, and this concern does not predict performance.

## Disposition and primary references

Bounded review complete: contributor coverage and reset placement check out under the stated contracts; publication proof remains unresolved. Stop here. Other scoped negatives, failed Factor-Y confirmation, closed BF16/attention/UNP endpoints and final-day lead permissions are unchanged. No exclusive phase share or removable-time budget is inferred.

Pinned implementation evidence: `tools/moe_short_ready.py`, `tools/moe_short_ready_harness.py`, `tools/moe_short_ready_host.py`, and `docs/steering/evidence/2026-11-03-moe-short-ready/paired.manifest.json` at73ab76cf. Archived resources show local strong GPU stores/acquire loads and retained fence counts, not target binary parity or the complete publication instruction span.

Primary semantics: [PTX8.7 memory consistency model, sections8.8–8.9](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model) and [CUDA12.8.1 warp synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions). These establish semantics, not a measured defect or speedup.
