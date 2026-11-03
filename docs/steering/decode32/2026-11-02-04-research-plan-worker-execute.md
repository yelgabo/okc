# D32-04: Research decides the design; Decode32 Worker implements and measures

November 2 Pacific / November 3 UTC, 2026. Evidence checkpoint: `a3ba392c724ec96de7b0cec7a6ae430506556d08`. The user has set Decode32 Worker's effort to medium and asks this research session to do the deeper planning so the worker can implement, send and measure. This changes the division of reasoning, not correctness requirements or the worker's end-to-end execution authority.

## Operating instruction — continue now

**okc research owns architectural reasoning, experiment selection and concrete implementation briefs. Decode32 Worker owns the entire execution loop: code, focused checks, private GPU submission, own-job result retrieval, analysis, bounded fixes, confirmation and D32 integration.** Do not send the candidate back to Driver or wait for research/captain approval at each step. Other project work remains with its existing owners. Public/scored permissions are unchanged.

Use the already supplied design rather than starting another literature review, architecture comparison, independent-review chain or general harness. Make ordinary coding decisions and fix localized implementation errors yourself. A specific unresolved numerical, publication, lifetime or progress contradiction is a real blocker; a desire for another generic review or proof of a future speedup is not. Publish a short concrete question for research only when a genuine design decision remains; do not bypass the technical requirement to keep moving.

## Fixed next experiment: finish D32-ATTN-01

The current implementation task is **within-request next-tile QK score carry for the L1-flat full-decode attention path**. Continue existing WIP; do not restart from a blank generator. Control remains public607 / `muse_decode_layer/cand/g1b4_stage_release.py`, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.

Follow the existing [source/lifetime brief](../../../archive/docs/steering/decode32/2026-11-02-02-driver-attention-source-feedback.md). The design choice is settled for this first experiment; its benefit is still a hypothesis:

1. Keep the current three-stage K/V producer, launch geometry, page-index handling, cache updates, half-merge and finalization. Do not toggle cross-request continuity or add another prefetch scheme.
2. Peel the first QK tile. Carry current and next score banks, targeting the existing eight useful FP32 scores per consumer thread rather than keeping redundant raw-K arrays live.
3. Interleave independent QK for tile `i+1` with tile `i` softmax/packing/PV work. Acquire the next tile with the original publication/join before reading its K. Preserve the existing two QK arithmetic chains and their reduction order.
4. Apply max, rescale, probability sum and output accumulation in the original tile order. Future scores must not update the current online-softmax state. Preserve all FP16/FP32 operations and the final BF16 boundary.
5. Retain ownership of current V until its last PV reader finishes. Keep the original release/join/fence protocol; use the existing `(tb+i)%STAGES` phase mapping. Drain lookahead at every request-segment end before the original merge reuses shared storage. Zero/one/two-tile segments and ring wrap must work without speculative out-of-range loads or divergent required joins.
6. Prepare a separate treatment module and select it only for the declared full-decode metadata family containing B32/q_len1/page1. Keep other standard cells on the control. All input values, routing, key coverage and unsupported-shape fallback semantics stay unchanged.

This is an implementation brief, not a new proof that arbitrary instruction reordering is safe or profitable. Verify these invariants in the actual patch, but do not repeat completed background research. A concrete contradiction in this schedule warrants one precise blocker, not a speculative replacement sweep.

## Execution checklist — no intermediate handoff

- Finish one generator/candidate and focused selector, bounds and slot-lifetime tests using `tools/d32_attn_worker_*` and `evidence/D32-ATTN-01/`. Reuse existing tests/readers; preserve notices and other workers' files.
- Run the relevant tests once, fix concrete failures, and rerun affected tests. Use an already available local compiler if practical. If this checkout lacks one, record the limitation and let the ordinary private judge check perform target compilation; a separate secondary-compiler relay is not mandatory. Do not claim a CPU mock established CUDA numerical or ordering correctness.
- Once the source is runnable and there is no unresolved correctness/progress contradiction, check live shared quota/jobs, publish the exact source, and **submit one ordinary private full-layer check yourself**. No prediction of a win, new review appointment or captain permission is required. Follow only your accepted ID; service/read errors do not authorize a duplicate POST.
- Archive source/job identities, actual selected treatment, official tests, fresh/cache/replay evidence and all twelve timings. Lead with D32 sliding, full, `C32 = 46*sliding_us + 16*full_us`, own baseline and all-other-load regressions. Use the existing current-service calibration `94ef698e91964dfa` as context, not as a matched causal control.

## Decide from the result

| Observation | Execute next |
|---|---|
| Local syntax/binding/indexing error with an identified fix | Repair that error within the same design; rerun affected tests. Do not redesign the algorithm. |
| Target fails numerical, cache, lifetime, progress or admission checks | Do not promote. Localize the concrete failure; only a documented implementation correction justifies another run, never blind retries. |
| Valid result has no useful D32 improvement, or overhead overwhelms it | Park this exact variant and publish the numbers. Do not start a tile/ring/warp sweep or build a confirmation harness for a loser. |
| Valid, promising D32 endpoint improvement | Run a bounded fresh matched confirmation with the existing infrastructure, independent mutable states/duplicate controls and cold replay. No new handoff or per-step approval. |
| Confirmed useful gain without unacceptable other-load regression | Integrate the D32-specific candidate, run full-layer regression checks, commit and report. Apply existing public-submission permissions separately. |

Twenty microseconds off full attention is a target budget, not a prerequisite or forecast. A cheaper smaller confirmed gain can be retained. Do not reimpose the old requirement that this single change improve the whole six-load score by 1–3%. Cross-job point scores alone are not confirmation.

## Current evidence to reuse, not redo

[MID32 result](2026-11-02-03-mid-result.md) is already terminal and parked; the inbox records C32 12.752280 ms versus the fresh control's 12.324230 ms. Driver's accepted-job exception has ended. Do not wait for MID, duplicate it, or rebuild its matched instrument. The attention source task was acknowledged at 23:55 UTC; that is not an acknowledgment of this newer instruction.

Keep reporting short: `source -> job -> correctness -> D32 timings -> next action`. Write one terminal result or one precise blocker rather than a new readiness report after each test. Record pickup in the existing status file and proceed; acknowledgment is not an extra gate. Research will use the committed evidence when asked to choose or revise the next design. This publication does not imply background monitoring, a new measured gain, or verified pickup.
