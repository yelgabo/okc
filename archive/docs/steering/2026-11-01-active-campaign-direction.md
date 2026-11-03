# Independent review of the active improvement campaign

November 1, 2026, 03:05 PST. Reviewed the active **Assess Muse Spark top-10 potential** chat, main through `169ac52`, and the decode-batch-one attention worktree through `ac64f99`. The assessment concerns the work actually underway. The [beyond-G strategy](2026-11-01-beyond-family-g.md) assumed a fully incorporated incumbent.

**Decision.** Continue the layer qualification and the standalone decode-one attention comparison. The worker independently found a stronger implementation, checked it on the official judge, and used score budgets to close two MoE ideas. The unresolved issue is the size of the layer opportunity. Qualifying a correct fork is necessary, but there is no demonstrated portfolio for the current top-50 cutoff.

## Whole layer

| Measured implementation | Owning job | Speedup |
|---|---|---:|
| Owned public entry | `cb71acec29d141c6` | 2.043438× |
| Exact upstream G0, off-board | `3a5f879df0d44825` | 2.248791× |
| Repaired fork with owned experts, G1a | `606f47f251f34873` | 2.010975× |
| Repaired fork with native experts, G1b2 | `736a5cbc977f4ad4` | 2.065559× |

The official leaderboard at review time placed the owned score at rank 378, rank 50 at 2.2867× and rank 10 at 2.3834× (CLI-rounded cutoffs). G1b2's point estimate needs another **10.71% score** for rank 50. It is 1.08% above the public point estimate on separate runs. The latest G1b4 fixes an empty-route access inherited by the earlier timed build via G1b3 and restores BF16 rounding before the attention output norm. It has reproducible source and CPU checks, with no H200 score. The layer had zero off-board checks left for the UTC day. [Qualification evidence](2026-11-01-family-g-qualification.md) and [per-load receipts](evidence/2026-11-01-top50/g1b2-shapes.md).

**First layer check after reset.** Run the already prepared G1b4 diagnostic. Verify native execution, private scratch and correct empty/occupied graph replay, all 13 official cases and 12 timings. Then require a clean pure-file check and a six-load comparison on independent matched seeds, beating twice the duplicate-control log variability. The owned control also needs the same pre-normalization BF16 boundary for this architectural comparison. Preserve the distinction between the historical public score and this control.

**Where I would sharpen the plan.** Prepare a discriminating performance experiment while waiting. The separate G1b2/G0 weighted latencies are 78.65/68.20 µs on decode 1, 288.27/258.67 on verify 16 and 378.77/350.80 on verify 32. Even substituting all three faster times and holding the other three loads fixed improves the six-load score by only about 5.6%. Start on decode 1, which shows the largest relative deficit of the three. On the selected G1b4 path, trace one producer-to-consumer edge and measure exposed setup or wait time. If legal, vary only when independent setup starts, with identical rounding, data, dispatch and fences, then time the complete graph. Record the resulting full-score bound before funding an implementation round. The endpoint differences identify where to look; routing, overlap, rounding and preparation changed between the measured endpoints. A conforming design still needs gains beyond recovering those three differences to reach rank 50.

## Standalone decisions

- **Attention.** An independent content search located a complete public implementation. Its exact source passed 16 tests at 2.639606× in `cd6805e7ebce45ee`; our public best is 2.604290×, rank 213 against a 2.7093× rank-50 point. Substituting its decode-one sliding/full medians yields about +2.00% standalone score on paper. The active worker is preparing a controlled comparison of two versus four QK accumulation steps/chains as implemented in the candidate, using the same BF16 boundary in both arms and checking the SGLang and official reference behaviors. This is a specific hypothesis about that discrepancy. Keep private input/cache state, held-out inputs and duplicate controls. [Source survey and scoring](2026-11-01-component-frontier-round.md).
- **Separate verify-one sliding screen.** Official status for `eac485ea0d554e50` was running at 03:05 PST. Its fresh-input loop shares cache state across arms, and its control and candidate differ in pre-norm rounding. Use it to screen the specific built candidate. A win warrants a private-cache, equal-rounding retest. A poor result can justify stopping the candidate under its budget, but cannot attribute the result to head ownership alone. [Screen limitations](2026-11-01-attention-complete-output-clusters.md).
- **MoE.** Correct to stop standalone scale compression alone on `05e51ddeb7014dd9` after the aggregate noise and regression gates failed. Trace `1c6ebc76b91d4b7f` gave an optimistic 0.34–0.43% six-load contribution for fixed-owner completion at the studied 128/256 loads. Reopen only with evidence for a materially different assignment or additional affected loads. These standalone outcomes do not settle the integrated G layer.
- **Projection.** The 30-shape comparison identified that several of the largest public wins used rounded BF16 split partials. The four-cell ideal substitution was +2.257% against an archived +3.356% target, before accounting for precision. Parking that dispatch sweep is justified. [Projection opportunity audit](2026-11-01-projection-weighted-dispatch-audit.md).

I inspected existing source, receipts, job status, available quota and leaderboard rows. The running screen and future G1b4 timings are open. No new GPU job was required for this review.
