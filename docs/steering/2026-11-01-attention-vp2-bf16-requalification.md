# Attention remaining storage lead: private BF16-output requalification

2026-11-01, root evidence af3b550. No GPU dispatch, public action, production edits or
foreign kernel adoption. Root subsequently authorized preparing ONE private comparator,
conditional on source/CPU/independent review. This checkpoint is not a qualified artifact.

## Source-backed opportunity decision

Rohan exact cd6805e7ebce45ee PASS16 gave2.639606424x (CI2.600012327–2.679200520).
Its D1 sliding/full4.673/10.794us versus owned historical5.183/12.390 suggests a cross-run
1.020034x six-load D1 counterfactual. Independent H200 runs/inputs and unrounded finalizer
arithmetic mean this is NOT matched BF16 causal headroom. Its n252/nc8/cooperative D1full
plan already exists in owned `_one_plan`; no new plan experiment follows. QK2, ring,
weighting, half-merge and distorted wait traces remain closed at their recorded scopes.

The distinct surviving lead is VP2 B1/q8/sliding: matched4862f8e0dc8a4b5a control11.10us
versus10.34/10.37us, independently built arms/two inputs. WSTfull16 lost and is excluded.
Saved VP2-only calibrated factor1.010861205 is an architecture-bundle signal, not an
owned gain. Failed f74cbe27819147c1 hierarchical DSM/global finalize and
eac485ea0d554e50 complete-head64-key/generic DSM do not implement its transport:
the latter uses dead raw-KV storage, requiring a post-drain cluster barrier before remote
generic pushes and a second publication barrier. VP2 uses a dedicated owner mailbox,
register-to-remote `st.async` stores and owner mbarrier completion; final reads are local.
Its N128 single-tile, one-head ownership, different query/P math and transport are bundled.
No isolated savings for async transport or a forecast for an owned rewrite is established.

The positive comparator also multiplies b0/b1 by inv then RMSes FP32 values. That omits
the SGLang attention-value BF16 operation boundary. Requalifying that signal with the
BF16 output boundary is the smallest actionable check BEFORE an owned N128/async rewrite.
It must not be confused with establishing every Q/K/P intermediate boundary: raw normalized/
RoPE Q/K still feed FP8 in VP2. Measurement only, distinct FP32-judge/BF16-output oracles.

Latest BF16-owned diagnostic3f44f891d0ef47db gives adjusted D1s/f5.491642/12.329775us,
V1s/f7.405782/14.541421us. Holding other loads fixed, sliding savings needed for1% six-load
are0.566813us D1 and0.722329us V1. Earlier VP2 ~0.735us mean margin is narrow and is not
portable across runs/precision repairs. Reproducible source pins/arithmetic are in
`tools/moe_attn_opportunity_audit.py` and evidence/2026-11-01-attention-storage-audit/.

## Provenance, license, scope

Private pinned keell0renz/muse-open-kernel-challenge@8e30e1be994db4310794b8d04e90d18d895deb84
file physical496d460e0bb1e7489684de2742f1ff22f3959d9d5fe30674199e74b3d002d651,
LF56c380b72001f88c1638c9315a02af4618b5af78263b1832ea6565e8676c95e4.
Its Apache2LICENSE/NOTICE and CUDA ZhimaoL attribution are retained in the private payload.
The direct upstream okc/x-vfgpdl branch no longer resolves; do not claim an independently
verified upstream commit. Auditor9cc027e/2c84ac6 archives that gap and mailbox proof.
No public standalone donor body is checked into Git: builder reads the private pinned cache,
and only manifest/logs/generator/owned harness are committed.

Builder `tools/moe_attn_vp2_make.py` currently emits409978B SHA
341132eb44919b1b258fc42f99bf7d3cc5a0f565fe0e51b5243186136fec3d0b.
This is a DRAFT pending actual host fixtures/independent final review; superseding hash
must be frozen before root dispatch. Four arms: two independently compiled owned ONE
controls and two independently compiled VP2s, four private namespace states/input/cache/
graphs. Candidate invokes VP2 ONLY B1/q8/sliding2048/page1/unitKVscale. All12 scored TP8
cells assert native owned control availability/plan/resource, all other treatment cells
actually call that control module with private state. Judged path is always owned control.
Unrelated PREP/ATTN/WS import-time builds are disabled; nonTP8 test inputs use existing
SGLang fallback, explicitly root-approved PRIVATE scope. All16 official tests remain mandatory.

Common owned ONE finalizers BF16-round value before RMS (both existing sites); VP2 b0/b1
RNE before RMS. No Q/K/P arithmetic or any fence/barrier/wait is edited. Public SCH is
forced0 in actual compile options, not merely its default/comment. Checked launch attributes,
zero-local and >=2 active16CTA clusters under the SAME launch config are fatal gates.
Actual VP2 launch bounds256threads/min1CTA, so no fictitious min2CTA claim; blocks/SM and
cluster admission both printed, never labeled achieved utilization.

## Mailbox ownership/lifetime audit

VP2 each source/owner transfers4heads*64*4B=1024B PV plus4 maxima+4 sums=32B,1056B total;
16 sources imply16896 expected bytes/owner, including empty partitions.1120-byte sender
stride separates payload/max/sum fields. Unique output/cache ownership tested by auditor.
Init fence + cluster arrive/wait establish destination readiness. Every owner waits for
all incoming complete-tx receipts before local reads and exits only after own receipt is
complete; no peer read survives another CTA's exit. Existing WGMMA waits/fences retained.
All-source empty partitions must still send every counted byte; future tests include short
contexts. No generation reused across launches: mbarrier/local mailbox reinitialized perCTA.

[PTX8.7 st.async](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async)
specifies register-source nonblocking shared-cluster stores and release complete-tx at
cluster scope. Destination data and barrier must be in the same cluster CTA; the current
8.7 text allows the executing CTA too. Do not incorrectly import older remote-only wording.
Acquire wait is required before consuming data; test success alone is not an ordering proof.
An owned equivalent four-row fixed-frame mailbox would use17408B +8B barrier, with16640
meaningful bytes/owner; it is NOT implemented or authorized by this source audit.

## Validation and stop gates

Local actual NVRTC/PTXAS12.8 sm90a VP2 BF16-output entry97regs/0stack/spills/2barriers;
owned8/15 entry83regs/0stack/spills/16barriers. These are compiler screens only; prior
H200 clean control resources differ, and runtime must report actual attrs/admission.
Device extraction adapters drop host driver/launch wrappers, use cuda/std/cstdint and
force SCH0; logs preserve initial failed header-adapter attempts locally. No host compile
or GPU claim. Compiler evidence: evidence/2026-11-01-attention-vp2-bf16-private/.

Two source/score tests pass: precision-only device delta, unchanged synchronization counts,
license/plan/resource scope, actual weighted-score function with winning/losing/NaN/JSON
round-trip cases. Independent actual TorchCPU full-host fixtures are pending. No dispatch yet.

Frozen intended cases: all12 cells x3 heldouts824731/824737/824743. Normal144/fresh336/
post144 checks; fresh means one bank for every cell/seed and17 distinct target banks,
each captured output/cache explicitly read after that bank's replay. Independent caches
and scratch, finite FP32 judge and separate BF16-output references/global/row/cache
thresholds; no thresholds weakened. Negative stale/nonfinite/transport/budget fixtures
must pass before source clearance. H200 startup must remain<300s; inner275s guards cannot
interrupt an in-flight compiler/kernel. Final summary<11KB and complete36rows required.

Timing uses32 rotated cold rounds, fixed3f44 per-cell raw-minus-adjusted calibration.
This is a score counterfactual, not an official candidate score. Six-load46/16 weighting,
all four crosses/every seed >=1% and >2x duplicate null, no>max(.5us,1%) regression.
No gain from an unmeasured cell or raw-only isolated VP2 timing. Any ordering/license/
resource/correctness/budget/coverage failure stops. On loss close exact requalification,
do not build/sweep the owned HC variant. On pass only propose the separately owned
representation/layout with full proof and a new independent authorization.


## Preflight refinement and real CPU math fixtures

Current draft6ca280a3b32f90f6781489f8af5dcf879d192a03659dfffcaf46854da8cea7b5,
410536B supersedes341132. Eligibility now includes page1/unitKV in the actual predicate;
legitimate unsupported page/scale cases delegate native control. Added16 untimed eager
checks (all4arms at contexts0/10/2047/2048), covering empty/partial source partitions.
Timing36rows unchanged; intended counters now eager_extra16/normal144/fresh336/post144.
Canonical CUDA device is explicit so preallocated private state key equals actual qkv.device.
Device strings/compile receipts unchanged. Two source/score tests still PASS.

Two additional actual Torch2.10 CPU tests PASS against EXTRACTED ACTUAL reference/checker
functions: distinct unrounded FP32/BF16-output oracles exercised, exact expected new cache
rows, and injected stale-output/NaN-reference/NaN-output/corrupt-cache negative controls.
CPU cache backing in these fixtures is FP32 stand-in for GPU FP8 indexing, while reference
new-row conversion really uses E4M3. This proves host/reference/checker behavior, not CUDA.
Command: hostvalidate Python fromtools `-m unittest test_moe_attn_vp2_math_cpu -v`.
Independent full36-case host facade remains pending; root must not dispatch this draft yet.


## Explicit wait-budget gap and logical byte accounting

No qualified V1sliding native wait-duration bound exists. Failed3f44 instrumentation is
full16/32, not V1; it cannot fill this gap. HC14.98/15.01us versus11.10us is an entire
architecture/precision screen, not the duration of its two cluster barriers. Source proves
two late blocking points exist; it does not prove positive exposed savings from eliminating
one or both. A trivial whole-kernel upper envelope is not an actionable wait budget.

For long V1 sliding, owned shape plan has33 nonempty64-key splits covering128 output rows.
Each contributes64 FP32 partials +one sum per row:1,098,240 logical global-update bytes,
plus66,560B final read/zero traffic (counter traffic omitted). These are payload requests,
not atomic transactions or HBM. Failed HC uses32 one-head senders*64rows*65FP32=532,480
logical DSM payload bytes. VP2 uses32senders*16owners*1056B=540,672 DSM payload bytes;
it carries explicit maxima as well as sums. Thus VP2 async transport does NOT reduce
payload versus HC. No measured cache-miss/DRAM/copy-latency conclusion follows.

Distinct storage/lifetime mechanism is dedicated mailbox versus dead-stage alias: dedicated
buffer permits early readiness setup, producer register pushes and owner acquire receipt,
then local-only reads without a post-drain raw-reader-alias barrier. N128/QK/P math remains
bundled. The justified private check resolves BF16-output-matched performance threshold
uncertainty (old ~.73us mean versus current calibration.722329us required), rather than
measuring a storage-only effect or spending an invented remaining-wait budget. If matched
signal fails, no owned rewrite follows; even if it passes, a later owned causal test must
separately establish transfer. No isolated barrier deletion is proposed.


## Pre-dispatch scoring registration and normal private comparator (supersedes earlier drafts)

Root explicitly authorized judging the foreign comparator PRIVATELY. Current artifact
78ce14e90f20cd3b406cd185d67d76e3624676739239d66032ff516e25a55d64,
411975B. Normal kernel_fn now uses the exported run closure on private owner2 with
record=False: actual VP2 only at exact eligible V1sliding, owned native on other TP8
cells, existing SGLang fallback otherwise. Official16tests/12benchmarks therefore qualify
the PRIVATE comparator, not merely its control. No public or adoption permission.
CUDA/math/ordering strings remain unchanged from the local compiler receipts.

Preparation separately asserts exactly20 VP2 and284 owned Python warm/capture calls;
actual graph replay calls are not host-counted. Coverage assertion completes before any
official calls. Thus legitimate unsupported-input fallback works but cannot silently
qualify an all-fallback private experiment. Actual plan rows identify VP2 grid32/cluster16/
threads256 or owned(splits,copies,mode), not a falsely attributed native plan for VP2.
Four independently compiled build IDs are included in the source manifest.

**Decision registration fixed before any H200 dispatch:**

- Preserve all36x4 raw times unchanged. Old3f44-offset gate and raw keep flags are
  separately descriptive; neither alone selects an owned rewrite.
- Derive12 raw-minus-adjusted offsets from THIS official comparator receipt, apply each
  offset equally to its four paired arms, reject any nonpositive adjusted latency.
- Require contemporaneous six-load46/16 factor>=1% and >2x duplicate null pooled AND
  every seed/all four crosses, with the frozen per-case regression gate.
- Also require all four RAW cross factors positive for every seed and no raw regression.
- Report actual official comparator score separately. Recalculated paired sensitivity is
  still a counterfactual scenario, not an actual official paired score or a storage-only gain.

`tools/moe_attn_vp2_read.py` validates the owning passing16test/12benchmark receipt,
complete coverage/dispatch/resources/plan metadata, recomputes raw and old-offset summaries,
then applies the registered contemporaneous-offset and raw-sign/regression rule. Two
synthetic-receipt tests PASS including independently different contemporaneous offsets,
flags/incomplete-backend coverage and nonpositive-adjusted negatives. Synthetic test
receipts are explicitly not H200 measurements. Source2tests and actual CPU oracle2tests
still pass; independent whole-host re-review against the final normal-path API is pending.


## Normal-entry ABI hardening and source clearance

Auditor identified a real eligibility blocker in78ce: config-only selection did not retain
original tensor dtype/stride constraints. New exact draft866845554b3d694c40165fabbdfde6739f6fa22b1efb5abb6b5831f32e368f24,
412952B restores strict metadata eligibility for BF16 contiguous QKV/shape, int32 page
columns with stride1/index bounds, int32 sequence, int64 contiguous cache locations,
FP8 contiguous K/V with matching shape/head layout, Float32 scalar scale metadata and
common device. Unsupported metadata delegates owned; an all-fallback private preparation
still cannot qualify because20VP2/284owned coverage is mandatory.

Actual source cleared by independent9bd60da; no remaining source blocker. Same CUDA
strings/ordering/precision/compiler receipts. Two source/score tests PASS; three actual
TorchCPU math/metadata tests PASS, including real Float8 metadata positive and wrong cache
dtype/location dtype/page stride/QKV/cache contiguity/nonunit scale negatives. This is
metadata testing only; no CUDA execution. The independent whole-host facade must mock
metadata explicitly for its FP32 cache arithmetic stand-in, then independently test the
REAL guard; the production guard is not weakened for fixtures. Final updated host review
pending. Root alone owns any H200 dispatch after final freeze and runtime gate review.


## Final source/host freeze

Final private artifact866845554b3d694c40165fabbdfde6739f6fa22b1efb5abb6b5831f32e368f24,
412952B is frozen; source guard/report commit e86b0d5. Independent source review9bd60da
clears ordering, ABI, metadata/fallback, provenance, mailbox completion and resource scope.
Independent updated whole-host fixture8/8 PASS, owner independently reran8/8 in7.396s.
Together with source2, actual CPU math/metadata3 and reader2 tests, no source/API blocker
remains. Whole-host math/CUDA calls are explicit stand-ins; real metadata and small dual
oracles/checker also execute independently. Resource/compile/capture/H200 math/time gates
remain mandatory and fatal. Root alone owns the ONE private dispatch/receipt. Frozen
manifest is in evidence/2026-11-01-attention-vp2-bf16-private/artifact-frozen.manifest.json.
No GPU result, public submission, owned rewrite or private-code adoption is claimed.
