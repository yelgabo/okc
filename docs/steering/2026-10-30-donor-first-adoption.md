# Donor-first adoption: test complete faster implementations before more invention

Date: 2026-10-30.
Our evidence snapshot: `5b88889a1ade991234d19ae2d2499e59e28425eb`.
Donor: `alygg77/okc-muse-spark-kernels` at `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`.

## Decision and correction

The user challenged why a repository with substantially better recorded standings had been treated mainly as a source of negative results and incremental ideas. The challenge is justified. The preceding source audit recommended against wholesale replacement before establishing whether the complete donor implementation is a better starting point. That was premature.

**Supersede that restriction: treat the complete donor files as replacement candidates. First establish their correct, reproducible performance against our actual releases. A whole-file replacement, donor-based fork, or selective hybrid may win; do not preselect our implementation as the base.**

Novelty is not an acceptance criterion. Shared ancestry with vjeux does not establish identical generated code, tuning, packing, scheduling, or complete-call performance. A failed implementation of a similar idea in our repository does not reject a different working donor implementation. Conversely, rank does not waive precision, memory-ordering, replay, or license obligations.

This report changes the next-run priority. It does not execute benchmarks, import kernel code, weaken any gate, or authorize a scored submission. Our main was unchanged since the previous source audit when inspected. No live leaderboard or quota query was made. The previous audit was a document/header review, not a complete source/SASS equivalence comparison or an H200 head-to-head test. This report does not claim to have completed those tests either.

## Evidence and why this is a high-value test

The donor README and RESULTS record a snapshot from October 29 at 20:54 UTC:

| Board | Donor score | Archived rank | Recorded job |
|---|---:|---:|---|
| Projection | 1.5502x | 75 | f07235a6ca8a46e2 |
| Attention decode | 2.6530x | 46 | 1c8d81a0dbd94ee0 |
| MoE | 2.4015x | 41 | e1bcad5a55cd4a29 |
| Full decode layer | 2.2058x | 40 | a61afb9687f14eff |

The donor was top 50 on several boards, not projection in that snapshot. This distinction does not invalidate the user's point. These are archived results, not verified current ranks.

Our last verified projection release is run `d85d0081fa8a4a43`, score 1.4975856117x, source `2265fa2`, submitted content hash `751bd89fa5bb314e8f57ade1e816e7b4aa09058720b00407de9a52d370c3acc0`. The frontier report records rank 314 at its October 30 CLI snapshot and a 1.5560x top-100 cutoff.

Nominal arithmetic: 1.5502 / 1.4975856117 - 1 = 3.5133%; the recorded cutoff needs 3.9006%. If 1.5502 reproduced under comparable conditions, 1.556 would be another 0.3741% in score. These ratios are motivation for a comparison, NOT a forecast, a causal gain, or a current leaderboard guarantee. The donor itself reports the same projection file scoring from 1.5259 to 1.5502 across four runs. Use matched comparisons to determine how much comes from code.

Useful target cases from the preceding audit, cross-run only:

| Case | Our six-receipt frontier, us | Donor best-run table, us |
|---|---:|---:|
| O8 | 5.88 | 5.52 |
| Pooler8 | 14.15 | 13.55 |
| Pooler16 | 14.72 | 13.54 |
| Router128 | 7.17 | 5.77 |
| Router256 | 9.67 | 7.35 |

Those five cases are leads, not a complete explanation of the aggregate difference. First measure the entire 30-case portfolio, rather than deciding in advance that only those five changes may be imported.

## Priority 0: reproduce the complete donor, not our reinterpretation of it

Each standalone owner owns this comparison independently; projection must not wait for layer M1 or MoE work.

1. Pin the exact donor file and verify its hash against the donor's `results.json` where available. Keep its source and license/NOTICE attributions. Freeze our exact released file separately from experimental main. Any hash mismatch is explicit, not silently accepted as the scored file.
2. Review eligibility against the local pinned task/reference and current published rules. Record specific precision or asynchronous-lifetime discrepancies. A documented correctness defect blocks promotion until corrected; generic suspicion does not justify skipping the comparison. Do not inherit a missing fence merely because donor tests passed. Preserve our validation requirements; adaptation required for a real defect gets its own clearly labeled arm.
3. Compile the donor as-is in the same judge-compatible environment. Avoid gratuitous refactors, inlining, tuning-table changes, or packing conversions before the first comparison. Host/API compatibility edits must be minimal, documented, and separated from performance changes. Do not substitute a supposedly equivalent rewrite for the actual donor file.
4. Compare independent preparations R1/R2 for our release and D1/D2 for the donor, with independent mutable workspaces, the same weights and input values, cold-L2 graph replay, and seeded balanced arm order. Use the existing corrected harness rather than inventing a new timer. Check output validity for every arm and record exact selected policies. Exercise fresh inputs and output retention; never cache input-derived outputs.
5. Cover all 30 projection cases and all 12 attention cases for their respective owners. Use held-out seeds and varied allocation layouts for robustness, not to select a favorable judged layout. Compute each problem's actual weighted load score. Keep raw candidate times as well as baseline-derived ratios. Independent duplicate controls must clear the claimed effect.

If an exact pinned full-file comparison already exists under these controls, reuse it and name the evidence instead of repeating it. A benchmark of an older vjeux pin or a subset of changed routes is not automatically this comparison. Confirm the donor attention file's body/hash against the previous reference: the donor says it is unchanged vjeux except attribution, but our currently shipped attention has diverged.

Budget: prioritize one bounded comparison run per problem, subject to actual shared availability, plus confirmation for a survivor. Do not spend all quota on a survey before testing the strongest available implementation.

## Priority 1: choose the better base, then isolate the remaining opportunity

| Outcome | Action |
|---|---|
| Donor is correct and consistently faster overall, with acceptable per-load behavior | Make a donor-based release candidate. Whole-file replacement is allowed as a technical outcome; keep the incumbent as rollback and retain our proven improvements where they add value. |
| Each wins different shapes | Build a deliberately bounded hybrid with shape/config dispatch, then benchmark the combined file. Include preparation/resource/layout interactions; do not just sum isolated deltas. |
| Donor gain vanishes under matched controls | Retain our implementation and report the per-load/control evidence. That closes this exact reproduction hypothesis, not every donor technique. |
| Donor fails a requirement | Identify the precise failed requirement and location. Test a minimal conforming correction as a new arm, or reject the affected path; do not quietly weaken correctness or portray the entire source as useless. |

An implementation need not beat the top-100 target in one jump to be a useful intermediate release. The previously adopted milestone check/repeat targets remain milestone checks, not requirements for accepting every reproducible partial gain. Scored submission remains the coordinator's responsibility under existing explicit user permissions.

For attention, "it is unchanged vjeux" is provenance, not a reason to ignore a reported 2.6530x implementation. Reconcile the exact full file, selected plans, generated code, required ordering, and our newer HMRG/PRQ or other changes. Do not revert our improvements blindly; test which combination wins.

For the layer and MoE, the archived 2.2058x / 2.4015x implementations likewise deserve a full-candidate path, not only extracting KPERM3. Use a separately owned comparison after static eligibility and quota planning. Do not transplant standalone FP32-router assumptions into the layer: its small-token activation precision has separate requirements. The current task, not the donor's README alone, decides conformity.

## What moves behind this decision

New codec GPU work, another dense schedule rewrite, and reopening failed QKV split-K should not consume the next comparative-test slot while the complete donor has not been evaluated as a possible better baseline. CPU-only feasibility and already-running experiments need not be discarded. Existing proven wins stay protected.

The donor's negative results still inform experiment design; they do not answer whether its complete implementation already beats ours. Likewise, our rejection ledger remains scoped to the exact tested designs. No need to establish an entirely new architecture before adopting a better implementation of a familiar one.

## Required handoff

The next result must answer: **What code should we use now: ours, the donor, or a tested hybrid?**

Commit the pinned file hashes, eligibility findings, per-case selected plans, raw/paired times and both duplicate controls, six-load score comparison, and a concrete adoption decision. Mark this report's comparison as pending until those artifacts exist. A new list of ideas without that decision does not complete this assignment.

## Sources

- [Our previous donor audit](2026-10-30-alygg77-public-kernel-audit.md), whose premature restriction on whole-file adoption this report supersedes.
- [Our projection frontier](2026-10-30-projection-top100.md), including the six-receipt table and timestamped CLI snapshot.
- [Projection run log](../runs.md) and [scored receipt d85d0081fa8a4a43](../../muse_bf16_proj/profiles/scored.d85d0081fa8a4a43.json).
- [Donor README](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/README.md), results, provenance, complete-file paths and retained license notices.
- [Donor per-shape results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/RESULTS.md).

The public rules page returned no readable text and llms.txt was inaccessible in this review; no new interpretation of contest rules is claimed. No new H200 measurement, complete binary equivalence proof, live rank claim, code adoption or submission is reported here.
