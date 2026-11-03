# CUDA optimizer and round-2 research: alignment, not an unverified verdict

November 2 Pacific / November 3 UTC, 2026. Repository evidence pin: `df237d362f67ae544473f8e30d6f5ba8b65755bf`. The user has let **OKC CUDA Optimizer** read **OKC CUDA research**, restarted the researcher for round 2, and asks whether the optimizer might be going down the wrong path.

## What this review can and cannot conclude

A narrow implementation-level CUDA pass can complement architectural research. It need not wait for round 2. But **this review has not obtained the optimizer's actual current plan, patch, selected source hash, or the complete exact CUDA research report**. It cannot approve or reject that agent's selected approach. Do not substitute the Decode32 Worker's plan for the separately named CUDA optimizer.

Discovery inspected the live main steering index, repository root/docs/steering path inventories, the first page of branch names, a scoped GitHub CUDA search, and available Project/Library searches. The searches returned historical material but did not identify the named current optimizer artifact or exact research report. This is a bounded access/result limitation, not proof that the work is absent, idle or wrong. Uncommitted local work and another session's current messages were not inspected. No claim is made about the still-running round-2 report.

No kernel, model setting, active assignment, GPU job, release, quota allocation or public-submission permission changes here. This note is not a new approval stage or a request to pause implementation.

## Evidence that should change how the optimizer uses the research

### 1. Pin the real target; similar names are not the same selected kernel

The retained layer source is `muse_decode_layer/cand/g1b4_stage_release.py`, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`, public job `607eb52bdaf24160`. The [native repair inventory](../../docs/steering/2026-11-01-g-native-repair-inventory.md) records tiny at T1, tiny2 at T8, MID at T16, B4 at T32 and MAIN at T128/256. Its actual MAIN definitions differ from a similarly named helper literal/comment. A suggestion must map to the selected source and instantiated flags, not a plausible-looking inactive function or a root filename assumed to be the release.

A real improvement restricted to T1/T8 can be useful for those loads; it must not be credited as a decode32 improvement. The user's batch32 priority remains important, but does not automatically convert this separately created optimizer into a second owner of Decode32 Worker's live edits.

### 2. Compiler-level changes can matter, but one major fix is already shipped

The same inventory documents a real loss of WGMMA overlap after activation-slot retirement repairs. Current-stage retirement restored overlap while retaining reader lifetime/publication, and its matched component comparisons produced factors 1.018493–1.028699. That recovery is already in 607. It is a good example of productive low-level optimization, not a pending opportunity to rediscover. A new scheduling proposal should identify a different remaining dependency or code-generation defect.

### 3. Resource metrics alone have already misled candidate selection

The [K128 terminal](2026-11-02-driver-k128-terminal.md), owning private job `bf8671b4bfa845a2`, actually admitted two resident CTAs and passed 13 tests/12 benchmarks/12 profiles, but scored 1.965158178x against the retained public 2.126082676x. Separate inputs/nodes/drivers prevent assigning the entire loss causally to occupancy or any single mechanism. The valid lesson is narrower: getting two CTAs resident did not establish a useful candidate. Fewer registers, fewer static instructions or more residency should explain a hypothesis, not substitute for complete-call timing.

### 4. Respect tested scopes without declaring whole technique families impossible

The native inventory records the direct/cached band32 locality attempts as closed at their tested scopes, SC6 retained in the G implementation, and a detailed observer whose overhead invalidated its attribution. Before reviving locality, ring depth, scale compression or instrumentation, identify the exact new source/mechanism/workload that changes the earlier result. Conversely, an old losing version is not grounds to dismiss a genuinely different working implementation.

Retain required BF16 boundaries, reader publication and progress contracts. Do not treat a passing numerical test or an upstream score as permission to remove synchronization. Question conservative implementation choices using a concrete sound alternative rather than calling all repair overhead unavoidable.

## Recommended working split, without another handoff chain

**Optimizer:** take one concrete, non-overlapping hot-path hypothesis from the research to a minimal patch and measurement under its existing authority. Record, in a short paragraph rather than another framework, the source/function/active shapes, what instruction/dependency/traffic cost changes, what prior result it differs from, and the complete endpoint being compared. Run focused checks; a runnable conforming experiment does not require proof of a future win. Confirm a promising result with existing matched infrastructure; park an unpromising exact variant rather than launching an indiscriminate sweep. Public permissions and shared quota/job ownership remain unchanged.

**Round-2 researcher:** supplement rather than block that work. For each recommendation, distinguish an observed current-code deficiency from a mechanism already implemented, a previous scoped loss, an unsupported hypothesis, or a new architectural design. Use the actual selected code, baseline and workload. Explain what would change the experiment choice; do not repeat a general CUDA checklist or assign a speedup from a paper to Muse.

**Decode32 Worker:** continue its own end-to-end lane without a new review dependency. The [D32 experiment](decode32/2026-11-02-01-full-decode-attention-experiment.md) and [ledger](decode32/STATUS.md) already record an attention candidate and accepted private measurement at the evidence pin. Do not duplicate that patch, poll/resubmit its job, overwrite its files, or reintroduce Driver handoffs. A separate optimizer finding can be recorded as an isolated diff; it is not a second concurrent edit to the same experiment.

## Smallest missing item for a concrete verdict

The optimizer's current plan or patch is needed: **exact source/function, affected workloads, intended change, and whether a job has already been accepted**. An existing message, diff or short committed note is sufficient. No elaborate report or new planning cycle is requested. Until that is visible, the correct conclusion is: the two-track setup is reasonable, but the actual optimizer direction is unverified—not approved, rejected, or known to be duplicating work.
