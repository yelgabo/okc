# H200 research intake and code application

November 2, 2026. Requested by the user in this chat. Initial evidence checkpoint `a3ba392c724ec96de7b0cec7a6ae430506556d08`; publication base `df237d362f67ae544473f8e30d6f5ba8b65755bf`. The [reference index](../research/cuda-h200/README.md) is the maintained entry point. The [implementation plan](../superpowers/plans/2026-11-02-cuda-research-application.md) records scope and gates.

## Result

Preserved the complete attachment, split its content into eight linked documents, mapped all 33 source sections and 36 recommendations, and added a checker for missing coverage, changed prose, broken local links and changed source evidence. Root `AGENTS.md` and `CLAUDE.md` direct future CUDA work to the index and relevant topics. Original source SHA256: `5cf54cd0f92975ab2347f9b32ec21b5817ada25caac82f137b946d62f2ca2f14`.

Applied a concrete repair to the validation workflow. Five MoE compiler wrappers previously hid compiler failure behind successful filtering or a later command. Both SASS wrappers also hid disassembler failure. They now return the failed tool's status, retain the full compiler log, and copy SASS only after success. [Before evidence](evidence/2026-11-02-cuda-research/before.log) reproduces seven hidden failures through actual shell entry points. [After evidence](evidence/2026-11-02-cuda-research/after.log) records the passing regression suite.

No runtime CUDA arithmetic, layout, dispatcher or submission source changed. This intake establishes no H200 correctness result or performance gain. Existing D32 execution ownership and jobs remain with Decode32 Worker under D32-02/D32-04. No agents, private GPU checks or public submissions were created.

## Reconciliation with our code

The uploaded research explicitly did not inspect the repository. Its advice needs the selected implementation and challenge contract before application. The audited control is [g1b4_stage_release.py](../../muse_decode_layer/cand/g1b4_stage_release.py), SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`, selected native body `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`. The separate file named `submission.py` cannot stand in for it. Retained owning receipts are public `607eb52bdaf24160`, fresh control `cc474883117947c0` and unchanged calibration `94ef698e91964dfa`. These are evidence identities, not a refreshed leaderboard claim.

B4/MAIN already use persistent scheduling, asynchronous raw-weight delivery, register-side BF16 FP4 decoding, RS-WGMMA, decode/compute overlap and fused nonlinear/finalization work. A decoded shared-weight bank would replace a working design. Tiny/MID need separate comparisons. Persistent BF16-expanded expert weights violate the task's storage rule, and FP8 substitution violates the MoE BF16 boundaries. Global post-dot scales and required rounding stay. The upload's barrier-removal branch requires a proved equivalent protocol before performance testing.

The [application ledger](../research/cuda-h200/application.md) marks each recommendation as present, applied, conditional, parked or incompatible. The [catalog](../research/cuda-h200/catalog.json) ties the notes to sections, prose units and evidence hashes. It detects identity changes; it does not prove semantic correctness or silently refresh stale hashes.

## Alternatives and stopping conditions

Compared the broad rewrite implied by the playbook with the current native engine and existing current Humming/FLUTE assessments. Their format, rounding, scheduler and net-cost gaps remain. The [grouped-work census](2026-11-02-worker-grouped-engine-work-decision.md) separates reduced allocated capacity from eliminated scheduled MMA work. A replacement must identify a distinct improvement and pay both GEMMs, packing, gather, SiLU, normalization and combine costs.

Retained exact parked implementations and reopening triggers. K128 full reuse `bf8671b4bfa845a2`, shared-A `5adf172986ff43dc`, direct weight `0b684a4065354fc5` and MID17..32 `a6fe2769779e4299` are terminal examples, not new allocations. Older phase-A Stream-K jobs `098949b28fe84bd1` and `e02f60fec94f45cb` price specific fixup/setup mechanisms. Skinny multicast census `53c07507b9fd4f8c` stopped at coverage before implementation. None rejects its whole design family.

The bounded implementation test here addresses the measuring instrument: failed compiler/disassembler commands must produce failure at the caller. It has a concrete result and needs no target quota. No new kernel candidate earned an experiment from the generic research alone.

## Verification and limits

- Twelve CPU tests pass: six checker tests and six wrapper test methods covering 23 wrapper scenarios. They cover source/evidence/prose drift, section omission, fenced comments, broken links, compiler/Docker/disassembler failures, first-of-two compile failure, quiet success and diagnostic retention.
- A real Docker CUDA 12.8.1 SM90a build and disassembly through the patched fused wrapper succeed for 11 standalone owned device entries. [The manifest](evidence/2026-11-02-cuda-research/device-compile.json) records source hash, image ID, flags and SASS hash. [The log](evidence/2026-11-02-cuda-research/device-compile.log) retains resources and existing spills. This is extracted device compilation, not the scored fork-native implementation or target host build. No spill improvement is claimed.
- Shell syntax and Python parsing pass. The digest checker covers all 33 real headings outside fenced code, eight documents and local links. Final counts and command results are in [verification.json](evidence/2026-11-02-cuda-research/verification.json).
- Primary architecture, compiler, profiler and paper references were opened separately. The upload's citation IDs lack their original URL mapping and remain labeled unresolved. No foreign code was adopted or executed.
- H200 numerical checks, Nsight captures, sanitizer runs, runtime occupancy and speed remain unperformed here. A fresh self-review checked notes and diff. Independent agent review was unavailable under the no-new-agents directive; the new claims have no independent semantic review.

## Priority, owner and next gate

Keep the active D32 attention experiment under its existing execution owner. This report adds references, not another assignment or poller. Future reuse starts with `python3 tools/check_cuda_research.py`, then the relevant topic and current steering state.

When the owner next needs a new discriminator, capture selected hot-loop issue behavior and physical DRAM/L2 traffic on legal production routes. Price a resulting treatment against the complete call and `46 * sliding_us + 16 * full_us`; preserve other loads and independent null controls. Stop before target allocation if there is no distinct limiting-work/dependency difference, if the build adds spills under the agreed gate, or if precision/publication/progress proof fails. Activity totals, logical bytes and peaks cannot establish endpoint savings.

Publication uses the isolated `codex/cuda-research-application` worktree. Only this chat's edits enter its commit. The shared D32 checkout and staging are preserved. Remote verification is a separate completion gate from local checks.

## Publication addendum

The concurrent [optimizer-alignment memo](2026-11-02-cuda-optimizer-alignment.md), published at `40bb040c`, was read during integration and retained. This library now provides the exact pasted playbook and its first-round application audit that the memo could not retrieve at its earlier checkpoint. It does not assess the separate CUDA optimizer's unseen patch or the still-running round-2 research. Both steering-index entries are preserved; this intake adds no gate to either worker.
