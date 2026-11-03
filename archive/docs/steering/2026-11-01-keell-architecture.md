# Keell0renz versus our architecture: visual source and performance audit

November 1, 2026. [Open the full HTML study](2026-11-01-keell-architecture.html).
The user requested an in-depth HTML study with visualizations of both architectures.

## Evidence boundary

Owned baseline commit `9aa3edfc18231784dc650782b5add184ae0f12ce`; upstream
`keell0renz/muse-open-kernel-challenge` commit
`8e30e1be994db4310794b8d04e90d18d895deb84`.
External layer bytes SHA256
`1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a`.
Windows checkout: `C:\Users\gretc\Documents\GitHub\okc`, remote
`https://github.com/yelgabo/okc.git`.

Read steering index, latest short-boundary and campaign-accountability reports,
new top-50 reset, coordinator method, context, ruled-out entries, current owned source,
all five external files and external README/LESSONS. Enumerated the pinned external
Git tree and local candidates. External source is retained outside tracked kernel code.

Existing external check `3a5f879df0d44825`: 13/13 tests, 2.248790532×,
95% CI 2.215058674–2.282522390. Our owning public layer receipt
`8f2780d7382047e6`: 2.022870×, 95% CI 1.992527–2.053214.
Recent unchanged owned-control profiles come from `35f2b5d66f51465b`.
The public/external score ratio is +11.17%; it is not paired causal attribution.
The HTML lets readers switch per-cell comparisons between the public incumbent
and recent owned control, and inspect all twelve pairs of kernel profiles.
Fresh compact board rows with owning public jobs are saved in
`evidence/2026-11-01-keell-architecture/boards.json`.

## Findings

Both lineages already have compressed projections, fused paged-FP8 attention,
persistent experts, asynchronous rings, shape dispatch, PDL and fused boundaries.
The architecture gap is specific schedules, formats, completion scope and dispatch,
not the absence of those broad ideas in our implementation.

The external WGS MoE uses weighted stream-K stage ranges for both phases,
four consumer warpgroups, six-bit lossless scales, and per-expert MFIN completion.
Owned mode 51 generally owns A units and claims B work. Our warp scans and
warp-parallel stage searches already exist; modes 74–79 already investigated
several donor-style finishing/range/epilogue policies. These are not new blanket
recommendations. Read the owning ruled-out evidence before reopening them.

SC6 reduces one weight stage from 36,864 to 35,840 bytes: 25% fewer scale
bytes but only 2.78% fewer total weight bytes. Bandwidth-only weight-traffic
speedup ceiling is 1.0286×. It cannot explain the whole layer gap alone.
External preparation logs report MFIN+SC6 bundle factors 1.0271/1.0329/1.0269
at 32/128/256 tokens. Both features change in the control, so their isolated
effects remain unknown. Its output-difference diagnostic inspects captured
outputs before replay; reported zero differences are invalid correctness evidence.
The official whole-file tests are distinct and did pass.

Recent owned/external verify32 sliding MoE profile rows are 215.6/197.2 µs;
full rows 202.8/190.2. But decode32 sliding is 79.5/81.7: external MoE is not
uniformly faster. Full verify32 attention is 185.1/186.7, which does not support
attention as that cell's main gap. These are separate-run profiler durations,
including possible PDL waits, not additive recoverable savings.

External projection dispatch adds rq at QKV128/256 and unpooler128 and pins
large-shape library plans. Its exact published projection scores are lower than
the team-best board row. Owned depth-two and push-epilogue variants already lost
in `4542b2b04cb344ec` and `4edaa20c98904404`; do not repeat those experiments.
Owned Lt transfer `1e3349e377234e89` passed a two-load screen; combined six-load
promotion remains separate.

External standalone attention actively passes `-DZH_SCH=1` at lines 5710/5722,
despite C++ default zero. The measurement-only piecewise-linear exponential
paths must not be confused with exact-math performance. The external layer is
separate and has no ZH_SCH switch. No standalone check was launched in this audit.

External prefill uses pinned cuBLASLt, N72 FMHA and persistent experts, with a
documented fresh-thread routing defect. Our checked-in prefill submission is the
starter and is not established as the source of our historical public score.
Prefill remains deferred; the artifact states this source-alignment gap.

## Concurrent progress and adoption limits

During this audit another session advanced through `60574a6` and prepared an
attributed fork. Its generator restores BF16 intermediate rounding, disables
deferred T1 glue, moves carry/tail loads behind dependency waits, removes the
invalid diagnostic and isolates attention state. See the live top-50 reset report.
These constraints mean the raw external score is not established as transferable.
A separate owned candidate was dispatched for scoring by that session. This
audit does not claim its result, alter kernels or authorize another submission.

## Priorities, validation and stopping criteria

1. Factorial MFIN × SC6 on identical data/preparation/codegen controls. Prove
   publication/acquire/reuse/progress before an owned implementation. Require
   held-out correctness and paired gains above duplicate noise, then score the
   complete layer. Discard any unsafe or non-winning implementation.
2. Finish the already-screened large-token library transfer and combined check.
   Do not replace it with another closed depth/push sweep.
3. Size short-path front/boundary consolidation in whole-layer score units.
   Top-k alone was already hidden in two T1 timeline samples. Require a credible
   2% layer opportunity or a small experiment that decides a named larger design.
4. Separate standalone attention exact math from scheduling before comparison.
   Confirm actual build flags and active routes, fresh graph outputs and score.
5. If prefill resumes, recover our incumbent's exact source before a fair audit.

No additive portfolio proves an owned top-50/top-10 implementation. Missing:
isolated SC6/MFIN contribution, matched all-source comparisons, corrected-fork
score, hardware counters/SASS resource attribution, standalone conformance and
historical prefill source. No GPU jobs, kernel edits or submissions in this audit.

The study is offline and self-contained. Its checked-in rebuild script reads
receipts and board evidence; primary source links pin both implementation commits.
