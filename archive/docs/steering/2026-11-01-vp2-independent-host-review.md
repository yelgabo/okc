# VP2 private attention comparator: independent host review

2026-11-01. Reviewer evidence base `1472655`; owner checkpoint `8f5155a`
followed by the owner-reported host eligibility/extra-context corrections.
No GPU, paid resource, registry change or public submission by this reviewer.

**Clear the bounded private diagnostic's host scaffold.** No new host blocker
was found in the final 410536-byte artifact with SHA256
`6ca280a3b32f90f6781489f8af5dcf879d192a03659dfffcaf46854da8cea7b5`.
This is not H200 runtime or C++ host-build qualification, a release decision,
or evidence of a speedup. Root owns any dispatch and its exact receipt.

## Scope and source attribution

Reviewed sibling `tools/moe_attn_vp2_make.py`, `moe_attn_vp2_body.py` and
`.build/today-moe/vp2_bf16_private.py`. The embedded manifest pins owned LF
source `0627edb55a4f28e8943c9e605246757287cde74f54ddc4306e0c8cca81457ea4`
and public comparator raw source
`496d460e0bb1e7489684de2742f1ff22f3959d9d5fe30674199e74b3d002d651`
from keell0renz commit `8e30e1be994db4310794b8d04e90d18d895deb84`.
The artifact includes attribution/license and is measurement-only, not donor
adoption. Its actual judged path remains common-BF16 owned native TP8;
non-TP8 uses the existing SGLang fallback.

The private treatment is restricted to B1/q8/sliding2048/page1/unit-KV and
16 query/2 KV heads. All other scored cells use the actual owned `_one`
path. Four namespaces have private resident buffers, QKV, caches, outputs
and graphs; there are two separately named owned and two VP2 module builds.
Selected native resident/resource checks precede timing. Duplicate resources
must match; VP2 requires zero local memory and the stated cluster admission.
Device compiler receipts remain the owner's separate evidence.

Both implementations receive the common BF16 attention-output boundary
before RMS. The independent FP32 judge-style reference remains unrounded at
that point, alongside a separate BF16-output oracle. The oracle distinction
is explicit; passing the BF16 oracle does not replace the judge-style gate.

## Executed CPU evidence

New independent `tools/test_vp2_private_host_cpu.py` executes the actual
host body through real TorchCPU tensors and AST-extracted actual owned
selection functions. Seven tests passed in 7.284 seconds:

- Whole `_vp_prepare`: all 36 timing records, 48 plan records, eager-extra16,
  normal144, fresh336 and post144 checks; four distinct resident states and
  private storage; only B1/q8 treatment; non-TP8 SGLang fallback.
- Actual captured closures read changed QKV; a deliberately frozen-QKV
  negative fails `VP2_MATH`. TF32 settings restore on success and failure.
- Actual checker rejects nonfinite output/reference and corrupted new-K
  cache. Actual score rejects nonpositive/nonfinite timings and exercises
  winner, loser, duplicate-null and regression decisions.
- Expired budget and nonpositive calibrated timing fail closed.
- A whole preparation with nonunit KV configuration makes no VP2 calls,
  proving actual host delegation rather than just checking predicate text.
- Actual FP32 and BF16-output oracle functions run on a small CPU attention
  case. Their BF16 results differ observably, and the actual checker accepts
  the BF16-output result against both references and the newly written K/V.

The whole-flow transcript was 6632 bytes, below 11000. This fixture uses
small CPU cache allocations, float32 cache storage and explicit stand-ins
for CUDA modules, events, graphs and attention arithmetic in the whole-flow
test. It does not simulate the published generator's full RNG consumption,
GPU FP8 accesses, partition behavior at real contexts, native admission or
elapsed GPU time. The separate small-oracle test executes actual reference
math; its cache storage is still a CPU stand-in. Synthetic timings test
score logic and transport only.

Reproduce from this worktree's `tools` directory:

```text
/root/.cache/okc-host-validate/bin/python -m unittest test_vp2_private_host_cpu -v
```

The fixture intentionally consumes the sibling generated artifact and body;
verify the exact artifact hash above before interpreting a rerun.

## Decision and stop criteria

Root may proceed with its one authorized off-board diagnostic after checking
quota and exact uploaded bytes. No separate reviewer dispatch is authorized.
Retain every real correctness/resource/fresh-state gate. Any unsupported
configuration must delegate to owned code, not broaden treatment eligibility.
The 275-second guards and fail-closed report budget remain mandatory.

Promotion requires every seed and pooled four-cross comparison to exceed
1% whole-score gain and twice duplicate null, with the per-cell regression
gate. The 46/16 load mixture and sixth-root calculation apply to all 12
cells; only one cell changes. Fixed earlier `3f44` offsets produce a
counterfactual adjusted score, not an official candidate aggregate. A host
or device failure is not a performance closure; a valid measured loss closes
this exact comparator experiment without an automatic knob sweep. No
shareability or adoption decision follows from this host clearance.

## Dated addendum: final normal-entry and raw-score revision

2026-11-01, superseding the artifact-specific clearance above. Final reviewed
artifact is 411975 bytes, SHA256
`78ce14e90f20cd3b406cd185d67d76e3624676739239d66032ff516e25a55d64`.
The owner changed host routing/reporting only; this review makes no new
device equivalence claim. Its normal judged entry now calls the returned
private run closure on arm2 with `record=False`: eligible input uses VP2,
other supported TP8 uses owned native, and unsupported TP8/non-TP8 delegates
to the original owned dispatch (SGLang where applicable). Thus the earlier
paragraph saying the judged path always remains owned does not describe
this final artifact. It is still an authorized private comparator, not a
public/adopted candidate.

The updated independent seven-test suite passes in 7.019 seconds. It calls
the actual normal entry wired to the returned closure, checks eligible VP2
raw-pointer dispatch/output shape, nonunit-KV owned delegation, and non-TP8
SGLang delegation. It inspects the actual closure's counter dictionary to
prove normal calls leave private coverage ONE284/VP220 unchanged. Whole
preparation with all nonunit-KV inputs now correctly fails the required
`VP2_DISPATCH_COVERAGE` qualification despite legitimate owned fallback.
The prior fixture treated successful all-fallback preparation as acceptable;
the updated negative expressly rejects that incomplete experiment.

All 36 report rows now retain RAW microseconds and 12 fixed historical
offsets. Tests reconstruct both the embedded stale adjusted gate and raw
gate exactly. The actual full-flow report still passes the 11000-byte
guard. Prior fresh-input, stale-capture, finite math/cache, timing,
calibration, budget, TF32 restoration and real-small-CPU dual-oracle tests
remain passing under the same explicit CPU stand-in limitations.

Reviewed the separate reader `tools/moe_attn_vp2_read.py`: it requires a
passing/unflagged official receipt, complete16 tests/12 benchmarks and
source/backend/plan/resource identities, then rebuilds the same paired
score with all 12 contemporaneous official offsets. The preregistered
decision requires that current adjusted gate plus every-seed raw cross
ratio greater than1 and no raw regression. Historical-offset and raw
1%-threshold keeps are descriptive. The reader is source-reviewed here;
no terminal receipt exists in this review and no measured decision is made.
Any actual nonpositive adjusted timing fails through the existing score
gate. Root retains final receipt validation and dispatch ownership.

### Subsequent strict metadata correction

The independent device auditor identified that normal raw-pointer dispatch
also needs tensor ABI eligibility. Final reviewed replacement is 412952 bytes,
SHA256 `866845554b3d694c40165fabbdfde6739f6fa22b1efb5abb6b5831f32e368f24`.
This supersedes the78ce artifact clearance. `_vp_eligible(data)` now checks
BF16 QKV shape/contiguity, int32 page/sequence metadata and stride limits,
int64 locations, matching contiguous FP8 cache shapes, scalar float32 scale
metadata, common devices and the exact supported configuration. Unsupported
metadata delegates to the existing owned dispatcher.

Eight independent TorchCPU tests pass in 7.507 seconds. The added test calls
the actual predicate on real CPU Float8 tensor metadata, then rejects bad
Q/sequence/location/page/cache/scale dtypes, noncontiguous Q/page tensors,
cache shape mismatch and unsupported configurations. The actual normal
entry with the real predicate routes the fixture's FP32-cache input to owned,
so it cannot reinterpret those pointers as FP8. For whole-flow arithmetic
only, the fixture explicitly replaces its FP32 cache metadata with equivalent
Float8 metadata before invoking the real predicate; full-flow cache arithmetic
remains the disclosed CPU stand-in. All prior coverage, fresh-state, score,
transport and failure tests remain passing. This is host-source clearance
for the bounded private diagnostic, with all earlier device/runtime limits.
