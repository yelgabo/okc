# Post-login experiments and recovered layer evidence, October 31, 2026 PDT

Evidence snapshot `fdaaa11`, following install checkpoint `158d3e8`. Checkout/remote remain
`C:\Users\gretc\Documents\GitHub\okc`, `https://github.com/yelgabo/okc.git`, branch main.
This report continues the [top-10 assessment](2026-10-31-top10-restart-checkpoint.md).

## Operational status

The user completed login. Authenticated jobs and quota reads succeed in Ubuntu through
`wsl -d Ubuntu --exec okc ...`. Before this task's dispatch each problem reported20 checks
and10 scored runs remaining. After dispatch attention has19 checks, other problems20;
these are shared account observations, not reservations. No public submission was made.
Saved account quota and official boards: [evidence](evidence/2026-10-31-post-login/).

Live layer rank373 at capture, score2.0187x, job `7b84ed4346d5497b`. The recommended
existing candidate remains `muse_decode_layer/cand/layer_m1rt51sc1g.py`; clean controls
and every submission.py remain unchanged. Top10 feasibility remains unestablished;
do not substitute a two-load donor ratio for a full score improvement.

## Recovered measurement removes a duplicate experiment

Authenticated job listing exposed completed check `0778baa78efa4aad`, which was missing
from this checkout's logs. Its receipt is now saved under `muse_decode_layer/profiles/`
with raw status, normalized profile, log and text. It is the exact d1/v1 current-build/donor
comparison proposed in the prior handoff, so no check was spent repeating it.

Passed13 judge tests; control score2.013476x,95%CI1.983274–2.043678. The two-load
four-arm comparison declares donor faster: pooled1.0768, crosses1.0693–1.0843, nulls
1.0132/1.0009. d1 pooled1.0836, v1 pooled1.0699. See
`docs/runs-decode-layer.md`, recovered-job section, for per-case times and source pins.

Crucial qualification: cand chose M1 `{8:o+front,16:o,32:o+front}`; cand2 failed its
fresh-input gate and chose no M1. Thus this does not isolate the donor's advantage over a
fixed M1 policy. d1 uses no M1 in either arm and the gap remains there. Named profile rows
show o+boundary5.5+5.9us against donor fused6.8us at d1 sliding, but the boundary includes
its wait. The4.6us difference between sums is not exposed latency or a proven fusion gain.
Do not reopen the losing cooperative C-1 form merely because that sum looks attractive.

## First new hypothesis: verify-attention address carry

Check `c77ce71c81af4fb1` is an off-board attention experiment, source `fdaaa11`.
The recorded opportunity is the0.4–0.9us per-request verify-loop residual from attention
round5, not a newly assumed full-attention bottleneck. New PRQ bits32/64 reuse the flat
loop's carried QK-descriptor and V-transpose address identities. Default remains31.
All existing fence/wait/barrier source counts are unchanged; V^T toggles after WGMMA
retirement. That static observation is not a SASS-equivalence or performance claim.

The bench has separately compiled/prepared duplicate controls, independent state for
three candidates (PRQ63/95/127), two held-out timing inputs,80 paired randomized rounds,
cold L2 and all12 scored shapes. It runs judge-style reference checks plus relative-output
checks. The official judged path remains the unchanged control. A winning arm must still
pass candidate-only official correctness and queued replay stress before promotion.

Status at this report's first write: running. No speedup, correctness outcome or resource
utilization is claimed for the new code yet. Bench source/hash/reproduction commands and
acceptance rule are in `docs/runs-attention.md`. Stop/revert if no reproducible gain beyond
null variation on the weighted score; do not promote a microbenchmark or headline control score.

Independent read-only source review of `fdaaa11` found no correctness issue: QK stage
offset is `st * 16384 / 16`; valid shared addresses remain below2^18, so no carry reaches
descriptor layout fields. Transpose load addresses match each original stage/key/lane/head;
XOR32 is exactly the old `2 * half` chunk contribution. Both V^T addresses alternate from
buffer0 as `it & 1` did, with toggles after P.V retirement. This is a bounded source review,
not replacement for official tests, replay stress or generated-code inspection.

## Correction to the MoE backlog

The prior checkpoint put variable-size x slots/deeper ring too early. A read-only review
of the latest run-log addendum (`docs/runs-moe.md`, after mode86's gate) finds an explicit
prerequisite: separate x-producer readiness from empty-slot waits first. Small jobs that
benefit from resizing are a minority at128/256, and mode86's earlier release saves<=0.6us
on normal routing. Increasing slot depth cannot resolve a copy that is blocked on readiness.

Mode56 receipt `7063f86352344efa` is verified against its owning file and log
`muse_moe_w4a16/profiles/bench_gskew51.7063f86352344efa.*`. ABGATHER reports first-ready
timestamps and ABSKEW reports consumer/copy timing, **not accumulated producer readiness
and empty-slot waits**. The omitted raw counters were not serialized, so this split cannot
be recovered at no GPU cost. Older41/44 counters combine terms and predate51.

Next smallest MoE measurement: a producer-only diagnostic twin of51, counting phase-A
xready, phase-B done_as, and empty-slot waits separately by phase, plus stage counts and
job-size buckets<=4 versus>4 n-tiles. Use separate preallocated diagnostic storage, preserve
consumer code and all acquires/fences/releases, and compare diagnostic perturbation with a
clean51/null twin at128/256 on held-out layer/standalone routing. This is attribution, not
a promotion test. Do not implement per-job resizing until there is an ordering/lifetime
argument across prefetched jobs, deferred WGMMA releases and barrier phases.

## Remaining validation and stopping criteria

Wait on the exact attention job, retain all breadcrumbs and log the result, including losses.
Candidate-only gates and paired confirmation are mandatory for any positive result, followed
by a full weighted-layer comparison. MoE producer instrumentation is next if attention's
address hypothesis fails. Never remove a fence, relax a correctness gate or submit donor code.
No measured optimization in this task yet supports the17.9% layer-score gap in the prior
top10 snapshot. Preserve the best candidate and distinguish bounded progress from a forecast.

## Completed result addendum, October31 PDT

`c77ce71c81af4fb1` completed in666.7s on H200. Official control tests16/16 and all five
arms' harness checks19/19 pass. The official2.524717x (95%CI2.486846–2.562588) is the
unchanged control's score. No experimental arm received candidate-only official timing.
All selected launch plans agree; no fresh register/spill/SM-utilization counters were measured.

**No performance promotion.** Raw cold aggregate ratios: QK carry1.0020, V carry1.0021,
both1.0004, identical control1.0008. Both's improvement is below null variation. The
single toggles' apparent score gains mainly occur on unchanged decode instances, including
d1 full's -0.26us also seen in the identical control. Affected verify shapes give:

| shape | control raw us | QK delta | V delta | both delta | null delta |
|---|---:|---:|---:|---:|---:|
| v1 sliding |11.17 |+0.06 |0.00 |+0.13 |-0.03 |
| v1 full |19.20 |-0.22 |-0.22 |-0.38 |+0.03 |
| v16 sliding |15.58 |+0.03 |+0.06 |0.00 |+0.10 |
| v32 sliding |20.10 |+0.06 |+0.03 |-0.19 |+0.03 |

v1 full both saves0.38us on each input; v32 sliding saves0.16/0.13us. v16 flips sign and
v1 sliding loses. These local readings do not meet the aggregate promotion rule.
ABSCORE is computed from raw graph medians, not official overhead-corrected candidate times.
An approximate reconstruction subtracting official per-shape overheads from rounded AB
medians gives both1.000519 versus null1.001257, preserving the decision. It is sensitivity
arithmetic, not a measured candidate score; `docs/runs-attention.md` records the full qualification.

Experimental source changes are reverted; the exact patch, log and status receipt (including
all benchmark/profile records) are preserved in `muse_attention_decode/profiles/` under the
job ID, and the implementation remains at `fdaaa11` in history. Loss/stop disposition is in
`docs/ruled-out.md`. No fence, precision, gate, production candidate or submission changed.

Next experiment remains the producer-only MoE diagnostic described above, not another address
toggle repeat. Best recommended layer candidate and top10 feasibility assessment are unchanged.
