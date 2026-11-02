# Qualifier inference repair and first H200 private-check registration

November 1, 2026, local Pacific time; registration after November 2 00:00 UTC reset.
Implementation began from main aff51efe21c8fe112bd89e1360ef76145375b032;
pre-registration evidence HEAD is 5705f88e3c0a527a91beacfeb5e9f25ad2e1d1fc.
Repository: https://github.com/yelgabo/okc.git. Sole implementation, measurement and
dispatch owner: this worker. Captain owns independent research and its separate report.

## Outcome and scope

The two immediate backlog repairs are ready for their first GPU measurement:
the inference-compatible all12 qualifier and its owning receipt reader pass all
32 CPU tests. Source regeneration is deterministic for both fixed arms. This
establishes CPU orchestration and source invariants, not H200 numerical correctness,
GPU progress, a preparation time saving, or performance. No new native math,
launch geometry, ordering fence, production kernel or reference arithmetic changed.

The builder imports the preserved b13ac0fd90dedc53a25edd5b1a739cd6103a1ea1
qualifier draft; the reader continues the preserved codex/today-attn WIP at
3a08ce4. Neither owner worktree was modified. The old failed97935e7d source
remains immutable. This is a setup/measurement repair, not an additional performance
candidate or a repeat of that exact failed implementation.

## Exact regenerated sources

| Arm | Path generated from committed builder | Bytes | SHA256 |
| --- | --- | ---: | --- |
| COMMON OFF | .build/rmw-v2/common.py | 918490 | 32196b8bc1c954e9b7f2c6667e6b3e0aeed14da0fed6fe53bc153c9d563e5047 |
| COMMON + short RMW ON | .build/rmw-v2/short-rmw.py | 918978 | 978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a |

Generate with Python tools/g_common_release_budget.py . OUTPUT --arm common
or --arm short-rmw. Matching manifests are archived in the evidence directory.
These pins supersede intermediate d39e2995/c60d9918 pins before compact phase
summaries were added. Reader accepts the two final pins and old97935e7d only;
draft822 and intermediate pins are excluded.

Private receipt support uses an explicit expected mode (--mode check); public
run remains the default. The narrow shared read/compare option retains all
source/job/log/terminal/content/coverage/score gates. Tests reject unexpected
modes for success, failure and comparison. GPU source pins are unaffected.

## Repair and source proof

Normal production prepare remains in the caller's original inference/gradient
context. Only private qualification runs under torch.inference_mode(False) and
torch.no_grad(), so its private tensor version counters exist. The exact-ID
back-reference memo retains front/weights identity, front/result/snapshot version
checks, exact IDs, finite checks, a four-entry bound and a fresh cell/prepare epoch.
It clears on success and exception. A hit still runs candidate-output/cache checks.

Flushed host phase/cell breadcrumbs and failure context preserve the last stage.
The final successful capsule carries phase counts, summed host phase durations,
whole elapsed duration and cleanup_begin/cleanup_end. The reader validates them.
These are host wall times including existing waits; no new device synchronization
was added and they must not be described as isolated device durations. Stable-ID
coverage gives379 events (331 fixed plus48 reference misses), bounded at475 if
all144 reference calls miss. The existing295-second guard and full validation
coverage remain. Two deterministic generations per arm match the physical files;
complete prior production prefixes and204 unaffected functions match the originals.

## Validation evidence

[evidence/2026-11-01-qualifier-inference-repair/cpu-tests.json](evidence/2026-11-01-qualifier-inference-repair/cpu-tests.json)
records exit0,32/32 tests passing in19.797s with Torch2.10.0+cpu under WSL.
The actual full prepare path is exercised for both arms, outer inference on/off,
grad on/off, normal/private failures, repeat epochs and cleanup restoration.
Negative cases include front/weight ownership, exact-ID mutation, saved snapshot
mutation, finite wrong output/cache after memo hits, nonfinite miss/hit/results,
native/resources/empty/fresh/graph/stale/guard/fallback evidence and malformed
reader transport/source/job/log/capsules. NaN JSON remains rejected as ValueError;
the negative test now accepts that existing parser exception.

[source-proof.json](evidence/2026-11-01-qualifier-inference-repair/source-proof.json)
pins code, complete generation manifests and source invariants. The actual reader
CLI successfully consumes the archived nonzero transport for own3fd1317143304bd5
as setup_failed, no score, no promotion and no causal RMW claim. See
[failed-observation.json](evidence/2026-11-01-qualifier-inference-repair/failed-observation.json).
Old3fd failed preparation at295s with0 tests/benchmarks/profiles; its expert-only
success did not qualify the whole layer. No new GPU result exists at registration.

## First private H200 check: fixed registration

User's latest clarification authorizes completing repaired readiness and registering
the first eligible arm without a mandatory separate COMMON GPU check or prior private
1% win. Select repaired short-RMW ON978b29f9 first: it can establish startup common
prerequisites and actual treatment exercise in one check, then run all official
correctness/benchmark/profile cells within the existing300-second preparation
limit. This maximizes first-check information without a new harness or weaker tests.
A complete single-arm result is descriptive and is not isolated RMW causality.

At November2 00:01:46 UTC the authenticated CLI reports layer5 scored/10 private
checks left,0 used after reset. These current caps differ from historical10/20
repo text. Latest50 jobs contain no live job; the newest is terminal3fd. Raw
quota/jobs envelopes are archived. Requery immediately before dispatch. Use only
okc check FILE --no-wait --json; root owns its returned ID and waits on that ID.
No public submission is authorized by the captain's messages or inferred from
start-work; any public measurement requires direct user authorization.

Hypothesis: moving only private references outside inference mode plus exact-ID
reuse allows the intact startup qualification to complete under295s. It may fail;
CPU tests cannot predict that. Secondary question: does the selected native short
RMW path execute with its required resource/progress contract? No gain threshold
is imposed merely to obtain this descriptive measurement.

Require exact final source/job/log ownership, actual H200 identity, clean terminal
success,13 official correctness tests,12 benchmarks and12 profiles, clean flags,
and native startup12 cells/72 model checks/12 captures/36 fresh checked replays,
60 empty checks/12 empty captures/48 occupied restores, five actual native expert
resource records and12 actual attention-dispatch records. Preserve legitimate
UNEXERCISED Marlin fallback as an official observation without RMW attribution.
Archive phase+cleanup times, raw/adjusted times, baselines/observer overhead,
source/resource/progress evidence and exact owning receipt. For each load compute
C=(46*sliding+16*full)/62; report all six and aggregate mean log ratio/geometric
factor. Historical607 comparisons are unmatched context only.

Stop on identity, numerical/cache/ownership, progress/capture/admission/resource,
setup budget or guard failure. Do not automatically retry or weaken qualification.
Retain losses/inconclusive results with receipts and scoped closure. No old gamma,
PDL, sparse or97935e7d repeat; no port of the unverified rohanarun lead.

The initial qualification/attribution tranche is capped at4 private layer checks;
this registration consumes at most the first. Do not spend the remainder until
this owning receipt chooses the next question. Promotion remains gated on fresh
independent matched OFF/ON states, both independently prepared duplicate/null
controls, all six loads, every held-out seed and all four OFF/ON crosses, and
regression checks. Existing
preregistered criterion: control/candidate crosses exceed1% and twice duplicate
log-null, with every context regression <=max(0.5us,1%), plus actual H200
module/resource parity or a reviewed admitted difference and all correctness/progress requirements. A single
successful official score cannot meet that causal promotion gate.

## Next action and unresolved gaps

Dispatch the registered ON check only after source/commit/quota preflight, then
archive and read its own receipt. No separate COMMON-first blocker. If setup fails,
phase evidence chooses the smallest next repair within this scope; if it qualifies,
bring the six weighted changes, aggregate log gain, actual resources and prepare/
cleanup timing back to the captain before the next attribution allocation. Captain
retains the research gate for the remaining G short-path cost/dependency budget.
The2.30 objective is captain planning, not a predicted gain from this treatment.

GPU outcome, module selection, elapsed prepare cost and performance remain unknown
at registration. Captain published593ba80 and1483aa1, then released README ownership;
this worker adds only the implementation link and preserves concurrent WIP.

## Accepted own private check, November 2 00:10:29 UTC

Registration and repair were committed/pushed before dispatch as
a0b6af2ec00dfed54f37fd81de00c14098748711. Immediate source regeneration and
quota/jobs preflight passed. The service accepted private check
**c1931f0c9e834485**, correct board muse_decode_layer, mode check, final
ON978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a/918978B.
Initial own status is queued/result null; its server content hash is
ce36aa2c06c6003cea3613d1e40d94f986d29cd38fc6352ccb7363b82aab30bd
and is distinct from file SHA256. Exact accepted envelope, initial status,
preflight and source association are archived in this report's evidence directory.

This consumes the first of at most four private layer checks in the initial
tranche. No public submission occurred. Sole dispatch ownership remains here;
an own-ID wait is attached. While queued, no GPU identity, startup result,
prepare duration, native exercise, official timing, score or causal effect is
established. Next gate is this own terminal receipt; no new arm or blind retry
is registered. The actual archived failure-reader CLI was rerun after the
explicit private-mode option and still reports setup_failed/no score for old3fd.

Latest observation 2026-11-02T00:13:17.236Z: ownc193 remains queued/result null.
The post-dispatch live layer balance is9 private checks and5 scored slots left
(one private used, zero scored used). No terminal result is claimed in this
accepted-job handoff; next gate remains own-ID retrieval and evidence review.
