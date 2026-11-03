# D32-06: continue the measured attention candidate; progress watch

November 2 Pacific / November 3 UTC, 2026. Evidence: `779b06ad6b5a1eefcd0a0006b6870668b987f2de`. The user asked research to watch this folder because the worker may be idle. The last recorded worker commit is November 2 at 16:32:41 Pacific. A completed commit is not evidence that the session is now asleep; local work and runtime activity are not visible through this repository read.

## Execute next — confirmation, not another architecture review

**Decode32 Worker: proceed directly with the already-authorized matched held-out confirmation of D32-ATTN-01. Do not wait for a new task, Driver handoff or research approval.** If this confirmation has already started locally or a newer accepted job exists, continue it and update STATUS.md; do not create a duplicate experiment or submission.

Keep treatment SHA256 `5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5` and control `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab` fixed. Do not tune the kernel while confirming it. Reuse the existing paired measurement infrastructure with matching inputs, independent mutable state/cache/scratch and duplicate controls, randomized arm order and cold replay. Share immutable compiled code where the existing infrastructure permits; do not share mutable state or change global selection between prepared arms. Avoid a new generic framework or review chain.

Use the frozen experiment's correctness, neighbor/fallback, attention output/cache and replay requirements. The inherited 56-replay banner covers experts, not a substitute for attention replay validation. Report actual selected treatment, D32 sliding/full and `C32=46*sliding_us+16*full_us`, duplicate spread and all-load regressions. Use the current D32 gate, not a legacy generic 1–3% whole-six requirement. Run the private check directly after live quota/job preflight and record its accepted ID; follow only that ID. Existing public-submission permissions are unchanged.

A confirmed useful complete-call gain proceeds to D32 integration and full regression validation. A valid neutral/loss parks this exact lookahead. An unresolved correctness/access/measurement-fit blocker is recorded precisely; do not silently sleep or replace confirmation with an unrelated sweep. Record the next execution state in STATUS.md as confirmation implementation, accepted job ID, terminal result, or a specific blocker. This status is a progress record, not another approval gate.

## What is actually measured

[First result](2026-11-02-05-attention-first-result.md), job `17383751c6a24010`: all 13 official tests, 12 benchmark cells and 12 profiles passed/returned; D32 sliding 149.764 us, full 312.070 us, weighted 11.882275 ms. Its own D32 baseline is 21.860480 ms and ratio 1.839755x. Whole-layer private score is 2.128829x.

Earlier control `94ef698e91964dfa` had C32=12.324230 ms, but different inputs/nodes. This is an exploratory comparison, not a proven speedup. The treatment changes full-decode attention only: changes in the unmodified sliding endpoint must not be credited to its QK lookahead. The first-run report already selects matched confirmation; this note reinforces that existing branch, not a new design. The original public 11.888-ms row is a historical reference, not a same-input control. MID and direct-weight experiments remain parked.

## Research-side watch

An hourly conditional watch was enabled in this research chat on November 2 at 16:37 Pacific. It reads current GitHub main, this inbox, STATUS.md and relevant subsequent records. Notify only on new results, concrete blockers, explicit waiting states, or no committed continuation across two checks when an authorized next step remains outstanding. Avoid repeating unchanged alerts. Do not infer that the worker is asleep solely from absent commits.

For documented waiting on an already-decided step, research can post a concise continuation note here without adding a handoff gate. This watch does not run GPU jobs, change kernels, claim direct session wake-up, inspect uncommitted local work or replace the worker's own desktop monitor. Monitor setup is verified; future runs and worker pickup of this note are not yet verified. No kernel, quota, active job or default release changed in this update.
