# G integration first: minimum control and ordered ablations

November 1, 2026. Root evidence checkpoint `c932a89`; common composition `2a6e3c3`;
source ledger `8cef427`; human-priority handoff
`ccb5e19`. This report supersedes the old stage-first/serial-front-second
queue in the 21:12 G integration handoff. The user's new priority is to
identify changes responsible for the public-G discrepancy before unrelated
projection, attention or owned-family tuning. Prefill remains excluded.

**No isolated price for the 5.77% historical score discrepancy has been
measured.** The source audit supports several required properties and some
conservative choices; it does not support charging the entire loss to
correctness. A minimum conforming upstream-based file is not yet qualified.

## What is measured

| File observation | Own job | Official score and 95% CI |
|---|---|---|
| G0, keell0renz `8e30e1be994db4310794b8d04e90d18d895deb84` | `3a5f879df0d44825` | 2.248790532, [2.215058674, 2.282522390] |
| Our stage-on G1b4, SHA `ba9e327c`, 869076 B | `607eb52bdaf24160` | 2.126082676, [2.094191436, 2.157973917] |

Both passed 13 official tests. Their ratio is 1.057715468. Inputs, natural
routing, timing adjustments and baseline measurements differ. The full
12-cell [owning-receipt comparison](evidence/2026-11-01-g-integration-gap/cross-job-gap.json)
remains the historical starting point. It is neither a paired regression
measurement nor a guaranteed recoverable speedup. The original author's team
board row is a third observation with a different job.

The largest weighted latency differences are decode1 (68.203874→78.549045 us)
and verify1 (103.618625→115.990823 us). Their native expert profiles are the
strongest investigation lead. Summed profiler categories may overlap and use
different routing; they are not exposed critical-path savings. Dense short
paths are already faster in that cross-job comparison.

At **21:23 UTC**, and again **21:38 UTC**, exact accepted job
`1fad70bbe3854cbc`, jobs and quota retrieval returned API530.
[Latest raw command receipts](evidence/2026-11-01-g-minimal-delta-plan/final-service-status.json)
retain the failure. The accepted job's outcome remains **UNKNOWN**. There was
no duplicate dispatch, new GPU run, candidate promotion or new score.

The broad goal is paused in the app. Its existing reset heartbeat is now
**PAUSED**, with the G-only priority saved and read back; its old ordering
cannot launch automatically. [Before](evidence/2026-11-01-g-minimal-delta-plan/reset-before.toml)
and [after](evidence/2026-11-01-g-minimal-delta-plan/reset-paused.toml) are archived.
The current work is the finite source/CPU investigation requested in the
newer human instruction. GPU continuation requires the human to resume it.

## Why upstream unchanged is not a qualified submission base

Attribution permission and Apache-2.0 provenance are already settled. The
remaining blockers concern the computation and synchronization, not authorship.

The [official activation rule](https://openkernelchallenge.dev/rules), our
owning evaluator's baseline, and the pinned SGLang interfaces require BF16
operator edges. G0 carries wider values across several of these edges:
gate/up→SiLU, scaled expert output→post-expert norm,
attention output→per-head norm/gate, and some boundary/pooler paths. A fused
implementation may remove buffers while preserving the rounds. Passing the
FP32-reference tolerance does not waive those interfaces. The
[precision contract](2026-11-01-g-fork-precision-contract.md) pins the
baseline/reference identity and primary sources; it also corrects the earlier
overstatement that every internal BF16 split partial is categorically banned.
G0 already rounds the SwiGLU output before the down projection; that boundary
was retained, not newly repaired, and supplies no additional-round cost here.

The native expert reader/publication protocol has specific missing generic/
async-proxy and reader-retirement obligations in G0. Original MFIN readiness
and DF sentinel implementations also cannot be enabled merely because their
old tests passed. The [ordering proof](2026-11-01-g-ordering-restoration-proof.md)
distinguishes required publication from the optional strengthening of the
final arrival atomic. Do not remove explicit fences or joins without an
alternative PTX proof and independent review.

Two **shared, inherited** interface/progress questions emerged in the new
minimum-control audit. They are not upstream→607 changes and cannot explain
that historical discrepancy:

- Selected ATT7 attention paths use software inter-CTA joins with ordinary
  launches. Instance occupancy checks establish a capacity bound, not the
  documented cooperative admission guarantee. Five historical cells are
  definite; two more depend on unreported runtime occupancy decisions. The
  polling watchdog traps, so this is not evidence of fabricated valid outputs
  or an observed hang. Resolve a launch-only alternative/proof before claiming
  a new fully qualified common control.
- The active pre-expert norm consumes raw BF16 gamma and forms
  `1+w` in FP32, while the baseline module prepares its effective coefficient
  in parameter dtype. The dense audit traced the actual baseline forward call
  to its cached BF16 effective coefficient. An OFF common repair is being
  constructed separately. Do not silently charge that common repair
  to the original integration loss.

CUDA's [block independence and grid synchronization contract, guide12.8.0](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#grid-synchronization-cg)
supports the admission distinction. Source findings establish proof gaps;
they do not retroactively establish a new measured failure in old runs.

## Complete source inventory and group boundaries

`tools/g_integration_source_ledger.py` checks both exact normalized source
pins and inventories every top-level AST statement without importing either
submission. It retains repeated `prepare`/`kernel_fn` occurrences and module
registration statements. Root's three provenance/coverage tests pass.
[Machine-readable inventory](evidence/2026-11-01-g-minimal-delta-plan/source-ledger.json):
402 G0 statements and 394 G607 statements; 329 align unchanged, 44 align
changed, 29 are removed and 21 added. These are source units, not 44 measured
optimizations. Whole native-literal hashes also include embedded comments.
Removed wrappers may only change prepare-time ownership or dormant paths.

| Group | Required property / chosen implementation | Attribution state and proposed separation |
|---|---|---|
| Expert precision | Required BF16 operator edges and effective post-expert coefficient; current helper applies scalar rounds in its data path. | No isolated cost. Preserve all rounds; test a mathematically identical precomputed effective coefficient as a separate representation if its complete consumer proof clears. |
| Expert shared/global publication and lifetime | Preserve actual reader/proxy fences, reader-counted release, CTA relays, coherent final reads and acquire. | No isolated cost. Unsafe removed-fence variants remain ineligible; a mandatory property's price cannot be estimated by toggling its correctness off. |
| Expert final arrival atomic | Current acq_rel RMW is one conservative mechanism in addition to retained writer fences and final acquire. | Proof-reviewed relaxed strong returning RMW is an eligible *source* alternative; exact existing f042 comparison isolates this group only. H200 qualification remains pending. |
| Expert progress and layer PDL | Cooperative expert admission is required for its grid-wait design; blanket PDLoff is a broader reference policy. | No isolated cost. Restore only a coherent edge with producer/wait/proxy/lifetime and progress proof; a global flag toggle is not an admissible ablation. |
| WGS-main and MID conservative choices | MFINoff/YBFoff affect main/B4; DF replacement affects MID/T16; SC6 retained. | Cannot explain both T1/T8 discrepancies. Existing fixed-owner128/256 bound only covers its measured scope. Keep behind short-path evidence. |
| WGS stage retirement | Later lifetime-preserving overlap improvement, already part of607. | Matched job63f5a2a8ed2341d1 measured factors1.018493–1.028699. It is already recovered performance, not a pending explanation for a newly added loss. |
| ATT7 output precision | BF16 RNE before value RMS/gate; native CPP otherwise unchanged. | Required edge; no isolated cost. Conditional SK2 fences do not execute under current PDL0. Do not assign their instruction cost to current execution. |
| Dense split/reduction policy | RQ disabled and opaque Lt split guard are conservative internal-reduction choices, distinct from required operator edges. | All eight607 Lt plans installed, so a rejected-plan fallback is not observed. RQ also has its own ordinary-launch software-join admission dependency if restored. No short-gap explanation or ready dense ablation. |
| Boundary/router ownership and rounds | Preserve BF16 branch/pooler edges, FP32 short router ABI, state-local scratch/prefetch lifetime. GLE_T/B1X are separate disabled rewrites. | No isolated cost. Retained FRT is shared withG0; its serial alternative is not an upstream→fork change. Do not reenable whole GLE rewrites via one wait/pointer fix. |
| Prepare/fallback/validation | Independent states, valid scale/layout selection, empty route handling and startup validation. | Classify setup-only changes separately from timed work. Preserve broad valid-input support; no timing-conditioned public dispatcher. |

The [native ladder](2026-11-01-g-native-minimal-delta-ladder.md), root `e132fd0`
(owner5958b28), reconstructs both actual native literals exactly: 33 ordered
source checkpoints covering12 transform classes and the subsequent stage
retirement/empty-route guard groups. The [dense inventory](2026-11-01-g-dense-minimal-delta.md),
root `86028e1` (owner87649fd), maps all12 cells and six contracts. The
[attention/host audit](2026-11-01-g-attention-mandatory-chain.md), root
`4dd7550`/`8e36c26` (owners8f43700/838f034), proves the exact pr8/PDL delta,
selected joins and the shared eligibility gap. Root regenerated all three
inventories and passed the native ladder's two CPU tests.

One OFF common attention host repair is implemented. DefaultOFF reproduces
exact607; opt-in emits870861B, SHA256
`dfac8955e5c45789b0cec40082498d75a133533253c84a5284db4c3dade80bbe`.
It changes only the launch for actual waiting finalizers, preserving selected
kernel/Params/grid/block/shared/stream and rejecting PDL/cluster combinations.
Root independently passed3 source tests, the exact host mock (two cooperative,
two ordinary,13 rejected cases), and compilation of L1/L8 adapters against
real local CUDA12.8 headers. [Root verification](evidence/2026-11-01-g-minimal-delta-plan/root-attention-host-review.json).
This does not link the whole Torch extension, run CUDA, prove graph capture,
measure actual admission/resources or qualify a submission.

## Working OFF code and exact source epochs

The [pre-expert common repair](2026-11-01-g-preexpert-gamma-off.md), root
`5527744` (ownerad3bf7c), separately prepares the baseline BF16 effective
coefficient and updates all seven Triton consumers/eight loads/15 call
arguments, preserving raw weights and the raw field. Its standalone enabled
source is869292B/4b3b4263. Root composed its disjoint changes with the ATT7
host repair in `tools/g_integration_common.py`, commit `2a6e3c3`:

| Provisional source epoch | Physical LF size / SHA256 | Qualification |
|---|---|---|
| Common control, explicit opt-in |871077 B / `4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280`|Source/CPU and independent review; H200 unqualified|
| Native post-expert gamma prototype, raw-format OFF |870703 B / `e17901d562b35f127c49110d222205fab254d078ae37e941a687774973a44c18`|Instrumented raw epoch; not byte-identical607|
| Native post-expert gamma prototype, precomputed ON |870702 B / `d6756bb50a0362af9c9cd03d44a88c80fa593628254ffb2a931a643fccafdcad`|CLOSED at local MAIN resource gate; no GPU candidate|

The common builder defaults OFF→exact607. Both leaf builders still verify
exact607 independently. The composition proves the pre-expert arm's ATT7
literal unchanged before replacing only that assignment; removing the exact
host repair recovers the entire pre-expert treatment. [Common manifest](evidence/2026-11-01-g-minimal-delta-plan/common-control.manifest.json).
The added8KB immutable pre-expert tensor remains retained in state; the old
generic private owner clone can copy it safely. A revised sharing inventory
must explicitly name it if sharing that storage between arms.

The [post-expert prototype](2026-11-01-g-native-gamma-precompute-off.md), root
`b43cbb5`/`56b0578` (ownersd7e10e1/71a647a), couples native slot7 to all32
gamma consumer expressions and all five compile/cache selector keys. Input
weights and existing Marlin fallback semantics are preserved. The OFF form
still contains the new format helper/selector, so baseline↔OFF compiled
equivalence is a required control before interpreting ON/OFF performance.
Its representation removes the old runtime addition as well as the inserted
BF16 conversion. There is no measured conversion-only cost or latency gain.

**Compiler stopping result:** baseline↔OFF normalized PTX, full normalized
SASS, resources and bounds are identical across all five fixed keys. ON
compiled at all five, but both selected MAIN entry representations increase
spills from4-store/32-load bytes to44-store/68-load bytes; registers96,
barriers16 and stack8 remain unchanged. Tiny/Tiny2/MID/B4 resources do not
worsen, including B4's pre-existing spills. This fails the registered
no-new-spill screen, so the exact full-familyON source is **CLOSED_RESOURCE**.
It is a local CUDA12.8 sm90a device-adapter result, not a new H200 resource
measurement or GPU performance loss. No GPU slot was spent, and no short-only
mask/adaptor rewrite or further sweep was performed. A future structurally
different format needs new source/ownership/compiler evidence. The factorial
below remains HOLD because its gamma arm did not qualify.
The [compiler closure](2026-11-01-g-native-gamma-compile.md), root `c932a89`
(owner18a17f1), preserves15 fixed build requests/30 entry instances,60
compressed evidence files and all manifests/logs. Root independently checked
all60 uncompressed lengths/SHA256 values, five baseline↔OFF hashes/resource
vectors and the two MAIN spill deltas. [Archive verification](evidence/2026-11-01-g-minimal-delta-plan/root-compiler-archive-review.json).
The loss is also recorded in `docs/runs-decode-layer.md` and `docs/ruled-out.md`.

Root's **16 new CPU tests pass** across the exhaustive source ledger, native
ladder, attention host adapter, pre-expert coefficient, common composition
and post-expert format/selector. Both coefficient checks cover all65,280
finite BF16 bit patterns; they do not execute the full norm or model on CUDA.
The [pre-expert/common review](2026-11-01-preexpert-gamma-independent-review.md),
root `afcd5b0`/`99b0212`, and [post-expert review](2026-11-01-postexpert-gamma-independent-review.md),
root `27f5cd2`, independently clear their stated source scopes. Exact root
commands/results are retained beside this report. No new full-model GPU,
resource, capture or performance measurement is claimed. These artifacts
are provisional controls/treatments, not a certified globally minimal fork.

## Ordered measurement plan

Every arm is a complete layer, with identical weights, inputs and natural
routing across arms, separate mutable scratch and graphs, and a common
conforming launch/interface policy. Static partial-patch checkpoints are
**not runnable candidates**. Never execute an unsafe partial repair just to
give it a number.

1. **Recover the already accepted job and current environment.** Own1fad
   terminal receipt, logs and profiles first; reconcile live ledger and quota.
   Record reported GPU/node/driver/runtime, compiler, image/evaluator and
   baseline identities; mark unavailable fields. Maintenance/node migration
   invalidates transferring the old point-score ratio as a fresh baseline.
2. **Freeze and qualify the common minimum control.** Resolve inherited
   attention progress and selected pre-expert coefficient contract with the
   smallest source change/proof. Apply it identically to both sides of later
   comparisons. Preserve stage-on607 behavior and every required edge. Run
   official13 tests/12cells plus fresh/capture/empty/ownership/guard/admission
   gates. Report its changed source separately from historical607/G0. Stop if
   a fully reviewed common control does not exist.
3. **Isolate the added final-RMW strengthening first.** The existing f042
   private check covers all12cells ×3heldout seeds, four owner placements and
   two variants. Its source/reader pins must be regenerated and independently
   reviewed after any common-control change. Oldf042 stays HOLD meanwhile.
   Preserve every BF16 round, explicit fence/acquire, join and finalizer load.
   Report every cell, null control and full-six result even on a complete loss.
4. **Post-expert gamma representation: closed before H200 at this source.**
   The proposed comparison was
   runtime BF16(1+rawBF16) against immutable prepare-time BF16(1+rawBF16) with
   direct effective-coefficient loads, only after full consumer/layout and
   scalar equivalence proof. This preserves the repair's numerical property
   while testing its chosen placement. Keep the RMW policy fixed for this
   comparison. Hoisting also removes G0's pre-existing runtime addition, so
   any win prices the complete representation change, not only the inserted
   BF16 conversion. The exact full-family prototype fails MAIN's local spill
   gate above. Do not dispatch it; preserve the negative compiler result.
5. **Factorial HOLD; test an interaction only after individual arms qualify.** If both
   representations remain eligible and measurable, use a fixed2×2 factorial
   of RMW and gamma placements with independent duplicate controls across
   all12cells. Do not add individually observed gains or select winning cells
   after seeing results. Current gammaON is closed, so this is not a queued
   GPU job. If an arm loses, keep that outcome and stop its sweep.
6. **Consider a proved PDL edge only if a material discrepancy remains.**
   First enumerate the exact producer/consumer and show that publication,
   waits-before-reads, parameter lifetime and resident progress all hold.
   Qualify one edge, then its interaction with the retained best source. If
   that proof is incomplete, report this portion unattributed and pivot to
   another compliant representation; do not restore globalPDL optimistically.
7. **Large-only conservative alternatives follow measured opportunity.**
   Revisit RQ/Lt or a new DF/MFIN representation only with resolved precision/
   progress proof and a named measured all-six budget. Existing closed
   fixed-owner, pending-flush, short-overlap, locality and XOR implementations
   remain closed. This is not permission for another knob sweep.

The old full-stage probe is **DEFERRED**: it requalifies a retained improvement
and is not the first integration-regression ablation. The old serial-front
probe is **DEFERRED**: FRT was retained fromG0. Old owned-four-cell work stays
HOLD; the accepted1fad receipt is still recovered for accountability, not
treated as a short-gap solution. New unrelated component tuning stays paused.

## Score, acceptance and stopping rules

For each of the six loads, form `(46*sliding +16*full)/62`; the official layer
score is the geometric mean of the six baseline/entrant ratios. Record all12
raw and adjusted latencies, official score/95%CI, flags, prepare time and own
job/source identity. For matched private comparisons the baseline cancels,
but raw cold GPU-event ratios are not official adjusted candidate scores.
The old f042 capsule retains medians of24 rounds rather than sample vectors;
do not invent a private CI from those medians. Preserve its complete loss and
resource records if a future revised experiment uses the same policy.

Attribution may record a small effect without promoting it. A candidate must
pass the registered complete six-load gate: every heldout seed and all four
control/candidate crosses exceed1% and twice duplicate log-null, with each
context regression ≤max(0.5us,1%). Require actual H200 module/resource parity
or a reviewed admitted difference, complete correctness/freshness/cache/
capture/guard coverage, and a clean pure-candidate official qualification.
No per-shape post-selection, microbenchmark-only promotion or transferred
historical baseline offsets. Compiler instruction/resource changes alone are
not latency or SM/HBM utilization measurements.

Stop dispatch while paused, API530 persists, own prior result/live ledger is
unknown, or quota/source identity is unverified. Stop an arm on missing proof,
precision/progress failure, failed resource/capture/guard coverage, prepare
budget failure, or a valid registered performance miss. Do not repeat a
closed arm without distinct new evidence recorded in `docs/ruled-out.md`.
If a conforming same-runner control cannot matchG0, publish the remaining gap
as unmeasured or requirement-constrained only at the proved scope; never
label the entire5.77% necessary. A recovered G0-like score alone would not
establish top10/top50.

Historical607 remains the measured performance incumbent. A new recommended
release requires the inherited common-control questions and new H200 gates
to clear. No new submission is recommended from source audits alone.

The machine-readable [ordered ablation plan](evidence/2026-11-01-g-minimal-delta-plan/ordered-ablations.json)
records exact source epochs, all12-cell coverage, independent versus
interaction questions, unknown measurements, legacy HOLD/defer decisions,
promotion gates and stopping conditions. It explicitly disables dispatch.
