# Sparse repair: standalone measurement and separate layer eligibility

November 1,2026. Starting main `ff05d88`; source implementation now integrated
as `94c3b86` (owner `79d5d2b`), independent format/lifetime proof `f3035fc`,
local helper GPU proof `d76c7bf`, full30 private harness `154ac09`, and actual
G applicability map `1839090`. The exact single-arm candidate and decision-policy evidence
are frozen at `c138969`. No H200 result for sparse repair exists at this
registration checkpoint. The prior turn was progress: it completed a working
dispatcher and proved the complete-owned verify hybrid still exceeds1MB.
That representation remains closed; this is a different projection algorithm.

## Live standings and feasibility

Official CLI snapshots at17:27UTC/09:27PST are archived in
`evidence/2026-11-01-projection-sk-xor-root/live-*.json`.

| Board | Ours | Rank | Rank50 point | Rank10 point |
|---|---:|---:|---:|---:|
| layer |2.1261×|316|2.2907×|2.4035×|
| attention |2.6043×|226|2.7208×|2.9331×|
| MoE |2.2157×|395|2.4579×|2.6077×|
| projections |1.5278×|317|1.5858×|1.6090×|

Overall was queried separately and puts us316; its API supplies ranks, not a
numerical speedup cutoff. These provisional observations do not establish
top50/top10 feasibility. The new algorithm has no measured speedup. The best
layer remains607eb52bdaf24160; best projection remains9eba96894ee1402e,
1.5278064829557674×,95%CI[1.504889385711431,1.5507235802001038].

## Evidence that changed the next experiment

The compression inventory found that C2 already uses a lossless12-bit stream;
the old context's statement that it was unbuilt is stale. Denser pair7/TOP7,
stage-depth and mailbox analogies remain closed. Instead, the actual SK source
shows a distinct dependency: separate escape warps read exceptional weights,
perform FP32 output corrections, participate in ring retirement and publish a
correction-buffer barrier that main consumers wait on.

The implemented variant keeps the existing12-bit payload and ten selected
geometries, adds sparse exact E7-XOR records, and restores original BF16 weight
bits in matrix-instruction registers. It removes the obsolete correction
readers/buffer/join. This is not an imported standalone implementation or a
denser-code retry. The C2/SGLang hotspot map was regenerated from owning public
9eba and baseline-as-submission bde8b4c49c4a46bd, archived in this report's
evidence directory. Profile activities are not exposed-component measurements.

The source prototype is310434B SHA256
`d23aa17be4cf9e7c5869803b3b4603d8e72395953687adc0287fb1c75738d586`;
native CUDA SHA256
`8cdbb4a99f226beb3f96ca662ae9d260bb7e002ef697337d8f80c075ccaa5068`.
It is off by default and is not the public measurement candidate. The private
full30 artifact is330402B SHA256
`e5d82cf2f1245495e070f6953a6f0b048cf21ac786b0cf4aa97384d3fc6f7020`;
its ordinary judged path remains C2 and it must never be submitted publicly.

## Precision, ordering and resource proof

Independent CPU interpretation restores every BF16 element for all ten actual
pack layouts, all65536 bit patterns, header boundaries,512-record units and
row maps. The exact helper also passed23 local RTX uint32 observations,
including20 graph replays with changed records. This does not validate Hopper
SK execution, projection math or performance. Local sm90a builds preserve all
ten control instruction streams and SX MMA counts without spills; candidate
code size, register changes and extra header/shared bytes are recorded, not
hidden. Actual H200 occupancy/admission remains unmeasured.

The source proof preserves all remaining main-reader fences. Headers are part
of the copied stage and transaction count; main consumers observe completion
before reading. Their header, packed-weight and activation shared reads finish
before the existing empty arrival. Global repair records are immutable. Only
deleted correction readers are removed from the expected arrival count; the
correction join has no remaining reader/writer. This follows PTX's transaction
completion, acquire-wait and async-proxy rules, rather than passing tests alone.
[PTX8.7 mbarrier completion](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier),
[async proxy](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-data-movement-and-proxy-operations),
[test/try-wait semantics](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait).

Restored BF16 bits do not imply bitwise equivalence to C2's old output-delta
accumulation tree. Independent FP64-to-BF16, row-error, exact one-hot, outlier
and changed-input graph checks are mandatory. We do not relax the C2 gates.

## Standalone decision and registration

The intended public measurement is a separate fixed, single-arm production
wrapper: one unchanged C2 preparation plus one SK extension, preparation-only
math/resource/fresh-replay checks, then fixed SX dispatch for the ten affected
scored cells and their ordinary bucket neighbors. Other20 scored cells and
M1 remain C2. No comparative benchmark, timing-based new-kernel selection,
measurement codec or private score machinery belongs in that public wrapper.
The built production file is321640B SHA256
`685ffab5d1614343d3c46d505967ae6817e3784736297018c78984354333f765`,
`muse_bf16_proj/experiments/sk_sparse_xor_release.py`. Its generator is integrated
at `69c8bb8` (owner `43a3261`), independent source review at `aff1f9e`
(owner `5a998f9`). Five actual TorchCPU host facades execute its preparation
and allM1..33/128/256/257 dispatch contracts. Runtime remains unqualified; an
ordinary successful H200 run must report10overrides,160mathchecks,10captures
and30fresh-replay checks before its score can count as an SX observation.

The latest quota query gives projection0checks/6scored, layer0checks/2scored,
MoE0checks/8scored, attention3checks/7scored. No cross-problem quota bypass is
used. Standing authorization for scored single-arm measurements can support
one projection measurement after source review; this is not a public layer
proposal or a claim that the new kernel already wins. H200 preparation must
finish within300s, with local295s guards; selected resources must match the
derived layout, have no local memory and admit at least the control's resident
blocks. Unsupported weight formats explicitly retain C2. Actual override
coverage is required before interpreting a score.

The separate private comparison is registered for three held-outs903151/
903157/903163 and24 cold replays: all30 cells, two rebuilt controls, two SX
treatments, actual C2 anchor, and actual common measurements for unchanged20.
Those common samples supply no independent null. Canonical whole30 policy is
`tools/proj_sk_xor_read.py`; at least1% full-score event gain and twice duplicate
log variability are required on every seed and against the C2 anchor, with
per-cell regression and anchor-drift gates. Same-receipt official offsets are
only labeled sensitivity calculations, never official paired candidate scores.
All42 combined CPU tests pass in17.451s, with the actual d23 packer, private
host/codec, immutable full30 policy, receipt validator and production
qualification/dispatch facades; full transcript is archived in this report's
evidence directory. No result, raw or adjusted, has been observed for this
experiment.

## Layer budget and stop/pivot criteria

[The actual607 dispatch map](2026-11-01-g-sk-xor-applicability.md) identifies
qkv/O/unpool BF16 interfaces at T1/8/16/32. G uses different SK2 geometry and
packing, so even a standalone win cannot be transferred as a layer speedup.
The short GLE_FRT pool/router requires FP32 router input; T16/32 PR is fused.
Current SX forbids those FUSE/HL paths. Verify128/256 Lt paths are unaffected.

On607's actual twelve medians, a conditional1% whole-layer gain needs1.800217µs
saved per short-load call in both contexts, or0.600072µs per qkv/O/un projection
if distributed equally. A proportional profile-time scenario needs7.3021%
removed from those three operators;5% gives only1.006820×. These are required
budgets/scenarios, not forecasts or hardware bounds. The830130B slim source
inventory is a concrete byte plan, not a built or qualified layer artifact.

Stop this exact schedule if correctness, ordering, admission or the registered
full-score/noise gates fail. No automatic tile/cap/codec sweep or promotion of
an isolated microbenchmark. Preserve C2/607; record every losing run. A public
layer experiment requires a separately built sub1MB file, native/fresh/model
checks and paired full-six46/16 evidence. Keep the existing first reset jobs
centralized and unchanged; do not duplicate them. No layer public slot is
selected by this report.

## Dispatch addendum: first official H200 observation pending

Root dispatched public single-arm job `2b537518091248f1` after all42 CPU tests,
independent source review, exact committed-file equality and live quota checks.
Source `c138969`, registration `1bd55da`, exact321640B/SHA685ffab5 above.
The CLI confirmed `muse_bf16_proj`; private checks0/public6 before dispatch,
no other recent live job. This is the production file, not the private harness.
H200 qualification, score and resource vectors are pending. No layer slot or
reset job was changed. Root is waiting on this exact job.

## Terminal addendum: qualified correctness, decisive performance loss

Owning public job `2b537518091248f1` completed on H200: PASS9 official tests,
all30 benchmarks/profiles, checks.ok and no flags. Preparation123.332824s.
Actual coverage is10SX overrides/0fallbacks/110pick keys,160startup math,
10captures/30changed-input replays. All40FULL/NF control/treatment resource
vectors passed the stated no-local/derived-layout/nonregressed-admission gates.

Actual score is **1.3941200708702308×**,95%CI
[1.3732082698071773,1.4150318719332842], below C2's1.5278064829557674×.
All ten target adjusted latencies regressed5.157–52.411%; raw latencies also
regressed7.372–35.141%. Other20 cells split10faster/10slower. The independent
six-load reconstruction exactly reproduces both official scores. Replacing
only the ten new adjusted values into the archived old20 gives a conditional
factor0.913729, not a paired score or causal attribution. Full rows, resources,
source/receipt pins and the normalization archive are in
[the terminal report](2026-11-01-sk-xor-release-terminal.md), integrated at
`a240b29` (owner `8bbaa7f`).

**Close this exact global-record/before-A production schedule.** No retry,
post-hoc subset promotion, direct layer port or replacement of C2 is justified.
The private global-record harness is preserved as evidence, not automatically
queued for another check on a closed production variant. Existing first G
reset jobs and the separate frozen four-cell verify comparison are unchanged.

A distinct stage-local record representation is being assessed, not selected
for a GPU run. Its source hypothesis copies the repair records in the same
bulk weight-stage transfer, rebases offsets inside that stage and derives the
arena capacity from actual immutable stage counts. It could remove the new
main-warp global-record dependency without reinstating correction warps.
The failed representation already leaves832–960B of rounded shared-stage
padding, but whether records fit is a pack/runtime fact; no chosen cap or
forecast gain is asserted. Exact inverse packing, byte/transaction accounting,
shared-reader retirement, unchanged controls, compiler admission and full-score
validation are still required. This is a new delivery representation, not a
rerun of losing bytes or an after-A/tile/cap sweep.

Safety does not establish free overlap: the before-A global-record chain can
hold stage credits. MT2/UPW2 late-arrive paths (qkv32/O32/unpool32) release after
MMA, so simply moving repair afterA is not universally after release. The
independent review's dated correction distinguishes an acyclic memory protocol
from this unmeasured backpressure. Actual counters/stall costs remain unknown.
