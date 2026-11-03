# Raw PR owned buffer: specific lifetime STOP, no candidate emitted

November 2, 2026. Allocation **d02a9f273a0e59a999946c344524877322144c06**, public owning-buffer research **f7763993d9504452b4d8b825eab7fa1e77fbe755**. Started 13:15:54 UTC; source/CPU stop gates completed 13:23:24 UTC, within 20 active minutes. This is the permitted specific STOP branch, NOT a working compressed allocator or source-qualified treatment. Serial remains parked; its archive is not reopened.

## Decision and concrete blocker

**STOP_MISSING_REAL_CALLER_GRAPH_RETIREMENT_CONTRACT.** The accessible preparation/kernel API and selected source do not provide a real-caller graph-owner registration or retirement point that proves no future replay/submission before completion and unmap/release/free. Installing a void owning tensor deleter and synchronizing preceding work is insufficient when a reusable external-pointer graph can still exist without retaining the tensor. No graph callback, global owner registry, process-lifetime leak or invented harness API was added to evade this requirement.

This is an inability to establish the mandatory lifetime proof in the accessible interface, not evidence that the actual judge has already used freed storage, nor a disproof of compression in a caller with an explicit lifetime protocol. The source/CPU counterexample demonstrates the missing guarantee, not a GPU failure. External/private runner destruction order remains unknown.

## Exact source and real-entry evidence

Base is unchanged public607 source muse_decode_layer/cand/g1b4_stage_release.py: **869076 physical LF bytes**, SHA256 **ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab**. No serial suffix, notice stripping or packaging change. Raw PR is 4608 by 8192 BF16, 75,497,472 bytes.

- Actual prepare at line 11132 constructs ordinary PR from BF16 pooler and high/low BF16 router parts.
- Actual pre-plan prepare at line **12450** calls its original parent before _graft_lt_plan. This is a concrete insertion point before both PR plan bindings; a post-prepare st.pr swap would be too late.
- Actual _graft_lt_plan at **12222** binds PR for T128 (tn, batch1, FP32 output) and T256 (tn, batch2, FP32 output). Each plan retains weight=weight, so plan/state references do preserve ordinary host ownership while they exist.
- Actual _graft_lt_run at **12272** passes plan weight.data_ptr as a ctypes pointer into cublasLtMatmul on the current stream. No graph reference or graph-owner object is returned by this entry.
- Actual final prepare at **17234** runs its captured normal preparation and original startup, then returns the ordinary state. The interface does not export a caller retirement callback. Internal startup/tuning graph scope is not proof of external timed graph retirement.

tools/g_pr_compressed_owner.py is deliberately a **fail-closed evidence gate**, not a VMM generator. Its build entry always raises MissingCallerRetirement after verifying the exact base. It emits decision.json only, never candidate.py. tools/test_g_pr_compressed_owner.py AST-extracts and executes the actual final prepare, actual pre-plan prepare, actual complete plan builder and actual Lt run entry with CPU-only mocks. It does not import the candidate, torch, CUDA or a driver library.

**Four CPU tests pass in 1.323 s.** Two independent states each have both PR plans pointing to their own PR; four interleaved actual Lt calls record the expected raw pointer addresses. Original preparation, prior-work completion stub and original startup run in order. After mocked prior-work completion and dropping one state/plan owner, its weak PR reference expires while the independently retained integer command list remains; the other state's owner is still alive. It also tests fail-closed build and pointer commands not being tensor owners. These mocks are not a CUDA graph emulator, ordering proof or target numeric test. The first attempt exposed a mock workspace alignment error (128 instead of required 256-byte steps); it was corrected in the mock only, and the initial failed log is retained. No device source repair occurred.

Accessible source scope includes TASK.md, AGENTS.md, the selected source, pinned owning-buffer research, git tracked paths and hidden/ignored rg path inventory. TASK documents one untimed prepare and timed kernel_fn(data,state); AGENTS documents CUDA graph replay. Neither inspected contract supplies graph retirement ownership. Filename-based harness/runner/judge/source inventory is archived with scope limits; a bounded search does not prove absence of all external harness internals. The pinned research already documents why foreign from_blob record_stream is not a retirement solution; this allocation performed no new external query or API experiment.

## Explicit non-implementation and unchanged inventory

Because safe retirement is a mandatory prerequisite, **no VMM host allocation implementation was emitted**. Added host bindings: none. Added Driver linkage: none. Modified CUDA/CPP/device kernels, ordered flags, math, fences, launch geometry, selected routes: none. Exact whole-base byte identity is stronger here than recapturing seven unchanged factories. No compiler, driver property query, GPU execution, okc/service call, check, public/scored/default action, subagent, workflow or new thread.

Capability/granularity/overflow/actual GENERIC property/device/access checks, independent bit-copy, nongeneric fallback, allocation/copy/plan failure rollback, and no-throw context-error cleanup are **NOT IMPLEMENTED OR TESTED**. No detached allocation-facade tests are offered as substitutes. Existing ordinary PR/plans remain intact because no treatment is installed. Target support, bit identity of a future allocation, compression ratio, capacity effects, eligibility, exposed time and speed are all unmeasured. Four passing CPU tests certify only these narrow source/lifetime gates, not the requested complete allocator success branch.

## Publication and next gate

[Owned evidence](evidence/2026-11-02-worker-pr-owned-buffer) contains decision.json, actual-source-entries.json, source-scope.json, successful and initial failed CPU logs, and generator/gate output. Its manifest pins both tools, this report and the evidence. README/shared ledgers and concurrent WIP are captain-owned and untouched. The final private check remains reserved; no target qualification is requested.

This closes this exact adapter attempt without a rescue loop. Reopening requires newly accessible, authoritative actual-caller ownership/retirement integration proving graph lifetime, no future submissions, completion and error-safe cleanup, or a separately justified different mechanism. Merely adding synchronize, tensor.record_stream, an output attachment, mocked close or a process-exit retention registry does not supply that missing proof. Captain owns the next priority decision; this STOP proposes no automatic follow-on.
