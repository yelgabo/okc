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
