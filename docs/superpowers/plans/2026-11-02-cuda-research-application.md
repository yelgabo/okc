# CUDA research intake and application

November 2, 2026. Owner: this research chat. Source checkpoint: `a3ba392c724ec96de7b0cec7a6ae430506556d08`.

## Goal and constraints

Preserve the user's complete H200 playbook, split it into reference notes, and reconcile every recommendation with selected code, task rules and previous experiments. Apply evidence-backed changes within this chat's scope. D32 stays with Decode32 Worker under [D32-02](../../steering/decode32/2026-11-02-02-end-to-end-ownership.md). Preserve concurrent edits. Create no agents. Public submission requires the user's existing authorization.

## Deliverables

1. Preserve the exact attachment under `docs/research/cuda-h200/`. Record its SHA256 and a section-to-note catalog. Keep original citation tokens as source data and provide separately verified primary references.
2. Write focused notes and an application ledger. Separate code facts, measured outcomes, hypotheses, incompatible recommendations and missing measurements. Link old losing implementations and their reopening conditions.
3. Add `tools/check_cuda_research.py` to check source coverage, links and pinned file hashes. Link the index from automatically loaded root instructions. A changed code file must flag its affected notes for review.
4. Repair failure propagation in the five MoE compilation wrappers. Reproduce compiler and disassembler failures through their actual shell entry points before the fix. Preserve full logs and publish SASS only after successful compilation and disassembly.
5. Run focused CPU checks and available CUDA 12.8.1 device compilation. Record the distinction between cross-compilation and H200 numerical/performance validation. No runtime optimization earns promotion from source inspection.
6. Publish a steering report and index entry, commit only this chat's work, push without overwriting concurrent changes, and verify remote bytes.

## Structural choice

A broad pipeline rewrite would duplicate existing persistent/TMA/WGMMA work, overlap D32 ownership, and reopen parked variants without new evidence. The bounded alternative is a complete applicability audit plus a real repair to the compiler evidence gate. This eliminates repeated generic tuning and prevents failed builds from appearing to pass. Remaining performance experiments need a selected-path gap and their existing owner, rather than automatic implementation of every suggestion.

## Completion evidence

The coverage checker must account for every research section and reject changed source evidence. Shell regressions must reject compiler, Docker and disassembler failures, retain diagnostics, and accept a successful quiet compile. Real device compilation is reported with its exact scope and result. Kernel functions, dispatcher defaults and scored entries remain unchanged unless a separately qualified change earns integration. Independent review is unavailable because the current project directive prohibits new agents.
