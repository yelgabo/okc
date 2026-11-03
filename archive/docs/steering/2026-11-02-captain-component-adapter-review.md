# Captain component adapter review and measurement handoff

November 2, 2026. Evidence base `c5a9d1284c01ae2fa2581ffb6b34173f2694098f`; adapter implementation `ece009b8`/handoff `6a97135b`; parent component-harness allocation `ca5cf8f0`; layer host integration `68f2f671`/`c5a9d128`. This is an independent review of delivered work, not a duplicate implementation or target allocation.

## Decision

**Retain the exact standalone adapter as a source/CPU survivor. No concrete source contradiction found in the reviewed scope. No dispatch authorized yet.** The next executable measurement path is the already allocated candidate-official component harness. The complete layer harness stays held. Top100 and any new incumbent improvement remain unproved.

## Independent evidence

Re-ran `tools/expert_top100_moe_test.py` from the shared main checkout using the existing WSL Torch2.10 CPU interpreter: **10/10 PASS, 1.618 seconds** as reported by unittest. Exact argv, return code and file identities are in [review.json](evidence/2026-11-02-captain-component-adapter-review/review.json); raw captured output is [adapter-cpu.txt](evidence/2026-11-02-captain-component-adapter-review/adapter-cpu.txt). This run did not install software or invoke GPU, driver, compiler, judge or public submission tools.

Read the actual adapter body, extraction boundary, compiler-call/entry tests, weight-bank/scratch tests and current standalone reference generation. Current ep8 input generation rotates the busiest expert group to0:32. The explicit local_start0/count32 restriction and first32 weight/global-scale slicing therefore match that current contract; this does not generalize to arbitrary shards. Unsupported ranges must continue to fail rather than silently remap IDs or change treatment. Fresh raw IDs/weights reach the native call, arm binding rejects crossing, and output allocation is per call.

The checked payload is `19aeaf356be6329dea1031966a4b7efbab4f039748f122acc78300e670cb5b57`,323514 LF bytes. Exact control `ba9e327c`/native `c7c201ae` and candidate `dff221ae`/native `3cc12ae0` remain unchanged. The tests exercise actual packers and original full scratch allocation separately, but full-preparation tests substitute small bank/packing leaves and recording native modules. Output stubs copy input: these tests establish host wiring/storage, **not an MoE numerical oracle or production-preparation pass**.

## Layer host repair review

Inspected the narrow runtime diff in `ca945bed`: capture guard state before original preparation; roll back to entry state on failed preparation/private work; retain official-state guards only on successful preparation; roll back again if final serialization, budget or reporting fails. It changes cleanup, not the expert numerical treatment. The published13 real-Torch tests and parent reproduction are retained evidence, not independently re-run or broadened here. Four complete wrapper/private-orchestration runs use real storage/inference/version behavior and CPU operator-tape replay, while production preparation, model oracle and CUDA remain explicit substitutes. Held layer source `4cdd4268`/918522B is not target-ready.

## Priority and next gate

1. Parent retains sole ownership of the bounded standalone harness allocated in `ca5cf8f0`; do not extend T1 or start another harness. The new wrapper must deliberately bind the **candidate** on the official judge entry, unlike the delivered adapter control default. Preserve two independent control and two candidate placements and exact native/build pins.
2. Review the delivered component payload/reader and actual generated-entry host negatives before allocating one private check. Candidate official correctness/replay, complete295s prepare fit, post-replay output checks, guards and strict owning receipt identity must qualify in that same job. No preliminary observer GPU job or automatic retry. Recheck fresh problem-specific capacity/no-live immediately before any later authorized dispatch.
3. Retain all fixed banks, equal-six whole-component >=1% gain on every bank/cross, gain above twice duplicate null and >=1% affected-load regression stop. A component loss can reject the exact treatment early; a component win only justifies considering a later complete-layer comparison. Never transfer component score, synthetic host timings or logical byte reductions into a layer performance claim.
4. Complete production prepared-state inventory and CUDA/target numerical/runtime/resource gates remain open. No hold removal, kernel/default change, public/scored action or new job is authorized by this review. This source-only acceptance has no owning GPU run ID and requires no fabricated run-log/technique-loss row.

## Gaps and stopping criteria

Stop before dispatch if the component harness is incomplete, changes native treatment/flags, returns control on the official path, lacks independently prepared states or cannot validate its real entry/failure cleanup. Any eventual target setup, identity, ownership, numerical, replay, progress or budget failure invalidates timing interpretation. Do not rescue by changing shapes, banks, gates or treatment within the same allocation. Other-thread delivery is pending; no completed component harness, numerical pass, timing result or measured win is claimed.
