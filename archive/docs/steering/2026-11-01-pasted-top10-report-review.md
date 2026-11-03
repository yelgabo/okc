# Exact pasted top-10 report: attribution correction and review status

November1,2026. Evidence repository source1764d15 and the user's newly attached text.
The user asked whether this specific paste had already been read. The correct answer is:
**the earlier documented audit covered a different report; this exact paste has now been
read in full.** Do not represent the earlier community audit as a completed audit of it.

## Verified source identities

| Document | Preserved evidence | SHA256 of normalized UTF-8/LF text |
|---|---|---|
|Earlier audited: OKC Community Audit Verification and Judge-Run Execution Plan|evidence/2026-10-29-community-audit/uploaded-research.txt|af3fb11f71a6def3a739edad29286e4dc61274915064cb3dd7e7558f8cb9e948|
|New paste: Reaching the Top 10 in the Muse Spark Open Kernel Challenge|evidence/2026-11-01-pasted-top10/reaching-top10.txt|f47bc1242e1a18e58afe82dc8d72bde6247f269e8771fa9781aedab029fffff0|

The original newly attached file is29155bytes, raw SHA256
6af1651787eeadd8afc378154c0814e72569bdfa2a3cf9f31530d9e3a9aa176f,
attachmentf162142c-7a96-44f6-af60-d7a468dc7dbe. The preserved copy normalizes CRLF toLF;
content comparison confirms the two research documents differ. The old file's working-tree
raw hash also changes under CRLF, so hashes are compared with explicit line-ending rules.

## What this exact report actually recommends

Its central recommendation is low-token cross-stage decode-layer optimization, especially
pooler/router/routing boundaries and expert-postnorm/unpooler/final boundaries. It favors
attention decode/verify and sliding/full specialization, persistent low-token MoE, and
limited further standalone BF16 tuning. This is different from the earlier report's
compiler/PDL/replay-audit-first priority order.

The report explicitly lacks our latest measurements and could not verify the official
leaderboard cutoff. Its8–18% layer opportunity, engineering-confidence ranges and projected
charts are estimates, not benchmark results or statistical confidence intervals. They are
not accepted as expected gains for our implementation. Its58% whole-graph example is a
public H100 profile; the report later explicitly identifies that hardware. It is not an
H200 measurement of our code. The report correctly states H200 bandwidth4.8TB/s; the prior
3.35TB/s correction concerns a separate repository plan, not this paste.

The pasted citation markers are internal research IDs without source URLs. They are not
independently verified citations simply because the report contains them. The official
rules/deadline/hardware facts have separate primary-source verification in our campaign
reports; each remaining external numerical claim still needs its owning source identified.

## Reconciliation required before treating this as an audited plan

1. Map every proposed fusion boundary to the current explicit layer candidate585d46e2...,
   its interfaces and stream/PDL dependencies. Mark already implemented work. A generic
   fusion recommendation does not establish legality or an end-to-end benefit.
2. Quantify the actual critical-path launch/traffic savings under the46:16 six-load score.
   Low whole-graph bandwidth utilization is insufficient evidence that activation round
   trips dominate. At low tokens, mandatory weight reads and overlap matter.
3. Check which attention specializations and MoE pipeline work already exist. Our current
   code already has distinct L1/L8/sliding/full plans and fused expert pipelines; the paste
   did not inspect that code, so these cannot automatically be booked as new opportunities.
4. Reconcile its inferred public rankings and gains with our current official snapshots,
   per-shape receipts, ruled-out experiments and measured resource gates. Preserve the
   original report's attribution; do not retrofit our newer measurements into its claims.
5. Keep prefill deferred under the user's latest instruction. Preserve existing candidates
   and all barriers/precision boundaries. Require official correctness, fresh replay and
   paired full-layer improvement before any layer-fusion promotion.

This is an exact-source identity correction and initial claim triage, **not a completed
external-citation fact-check or a measured fusion-speedup claim**. The next audit work must
explicitly cover this preserved text rather than substitute the earlier community report.
See [current four-track execution](2026-11-01-four-track-progress.md) for the tested code,
public scores, pending experiments and unchanged top10 gaps.
