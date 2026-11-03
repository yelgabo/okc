# D32-ATTN-01: Decode32 Worker — implement one full-decode attention candidate

Date: November 2, 2026. Research assignment following the user's explicit creation of **Decode32 Worker**, direction to use this watched folder, and clarification that the worker is waiting for tasks. Evidence: inbox setup `a4192dec`; D32 priority `b9ca9dff`; captain's fixed MID allocation `5b4a729a`; retained public layer `607eb52bdaf24160`. Read newer captain commits before touching shared work. This file is an actionable implementation assignment, not a claim that pickup or a performance improvement has already occurred.

## Start here — ownership and deliverable

**Decode32 Worker owns the separate D32 full-context attention implementation track. Start source/code work on this task; do not wait for the MID result.** OKC Developer retains the existing MID-versus-B4 expert task. OK Driver remains the sole judge dispatcher and integration owner. This partition prevents two workers rebuilding the same MoE candidate. Do not alter or cancel the other owner's files or jobs.

Deliver **one reproducible plain full-layer candidate**, focused tests and a short handoff for Driver to measure, or a specific source/correctness blocker with the actual failed mechanism. A general literature review, another folder setup, or an unchanged source audit is not the requested endpoint. No new public submission or production-default change is authorized here. Judge measurement remains Driver-owned; do not call `okc check`, `submit`, `withdraw`, or poll another owner's jobs.

## Objective and exact scope

Improve **decode batch32, T=32, q_len=1, full context, page1**, retaining every key and the complete layer interface. This is not verify32/T256 or q_len8. The user's weighted load is `C32 = 46*t_sliding + 16*t_full`, approximately 11.888ms against 22.149ms baseline (1.8632x). The first combined MoE-plus-attention milestone is 2x; it does not have to come from attention alone.

Aim for a meaningful net reduction in the complete full-context call; **20us is a development target, not a forecast or minimum correctness gate**. It would remove 0.320ms from C32, about 2.69% of that load with other times fixed. A cheap, repeatable smaller improvement can still contribute. Do not impose the obsolete requirement that this one attention change alone improve the six-load layer score by at least1%.

Use the retained `muse_decode_layer/cand/g1b4_stage_release.py` (SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`) as the explicit control unless Driver has published a newer qualified control. Its fresh `cc474883117947c0` profile reports about171.4us for D32 full attention; that is activity duration, not measured removable slack.

## Concrete work

1. Resolve the actual selected L1/full path (`attn_kernel<1,true,0>` in the retained profile), its preparation/launch/merge/cache interfaces, and current closure evidence. Briefly record what work and live storage a changed schedule would remove or overlap. Reuse the existing source and receipts; do not build a new general profiler or benchmark framework.
2. **Preferred implementation hypothesis:** overlap independent next-tile QK/data movement with the current tile's softmax/PV work in the actual single-query flat path, rather than merely rearranging instructions within a completed tile. This is a proposal, not a verified untried mechanism or promised win. Inspect prior cross-tile/continuity results before implementing: a label alone does not make a previously failed schedule new. If that exact mechanism is already implemented or specifically rejected, select one materially different complete full-decode path supported by the source comparison and explain the difference. Keep this to one candidate, not a tile/warp sweep.
3. Implement in an isolated generator/candidate. Preserve other standard load paths, MoE choices and input preparation. State the supported metadata family containing B32 and retain the control for unsupported shapes/lengths/scales. No seed/value-based selector or cheaper routing. Keep current-cache writes, causal bounds, FP8 cache representation, permitted query precision, BF16 output boundary, gate/norm and returned-state semantics.
4. Prove each new buffer's producer, final reader and release event. Preserve required proxy/publication fences, progress/admission and graph-replay ownership. Use existing CPU tests and the existing local compiler when available; inspect spills, shared memory, instructions and admission as evidence, not a promise of speed. Do not install a toolchain or invent target resources. Actual H200 qualification remains open until Driver measures it.
5. Publish the exact candidate hash, generation command, focused source/selector/lifetime tests, observed compile results, required target checks, and the one changed mechanism. Give Driver an ordinary full-layer measurement path and existing receipt-reader instructions. No proof of a speedup is required before proposing that bounded exploratory run.

## Avoid duplicating completed work

Read `../2026-11-02-worker-attention-continuity.md` and `../2026-11-02-worker-attention-continuity-reconciliation.md` before touching the old continuity prototype. It already has a prepared second module, isolated L1 merge scratch, initialization join and a source-tested selector. Its broader proposal was held for ROI, not measured as a win. The report warns that at most31/264 CTAs cross a request boundary at B32; do not equate that continuity change with whole-loop overlap or claim20us from it. Reuse valid infrastructure rather than rebuild it; do not overwrite the previous owner's generator.

Do not repeat standalone scalar/address tweaks, q_len8-only work, the failed direct-weight/K128/shared-A expert variants, or the MID task already assigned to OKC Developer. This D32-specific assignment supersedes the earlier general attention deprioritization only for this isolated track; it does not reopen all attention experiments.

## Files and handoff contract

Own new files under `tools/d32_attn_worker_*`, an isolated candidate/evidence directory under `docs/steering/decode32/evidence/D32-ATTN-01/`, and dated reports in this folder. Check for existing ownership before writing. Do not edit the incumbent, shared run logs, root instructions, the MID generator or another worker's WIP. Commit/push source and handoff without force-pushing; retain attribution and the submission-size limit.

Record pickup in `2026-11-02-01-full-decode-attention-status.md`: taskID, worker, source pin, owned files and actual start. On completion update it with candidate/report/commit or the concrete blocker. Publication and monitoring configuration are not pickup; numerical or compiler success is not measured performance. If the current captain has already allocated this same attention path, reconcile that overlap rather than duplicate it.

For any eventual measurement Driver reports raw/adjusted D32 sliding/full times, weighted C32, own baseline, actual selected path and all-six-load regression results. A favorable ordinary run earns matched held-out confirmation with independent mutable state and cold replay; it is not automatic promotion. A broken math/lifetime/progress requirement blocks the candidate. A valid losing candidate is archived at its exact scope, without automatic rescue sweeps.

## Evidence and next action

- [User D32 objective and weighted budgets](../../../../docs/steering/2026-11-02-decode32-priority.md).
- [Existing MoE MID assignment — other owner](../../../../docs/steering/2026-11-02-driver-d32-mid-comparison.md).
- [Continuity implementation and unresolved target gates](../../../../docs/steering/2026-11-02-worker-attention-continuity.md).
- [Fresh incumbent profiles](../../../../docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/hotspots.md).
- [Layer task and numerical/scoring contract](../../../../muse_decode_layer/TASK.md).
- [Inbox ownership and monitoring convention](../../../../docs/steering/decode32/README.md).

**Next action: acknowledge this task in the status file and start the isolated D32 full-attention candidate. No need to wait for another research assignment.**
