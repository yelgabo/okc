# Lane3 final source screen: scaled expert-output boundary

2026-11-01. Requested bounded paper/source follow-up to paired-RNE closure
`aafc32d519efddab942febe5fe564796604ea618`. No further representation prototype,
compiler variant, GPU execution, official/service call or production change.
Root owns index/integration. **No independent qualified lane3 candidate emerges.**

## Exact changed group and source evidence

The selected alternative group is the short expert GEMM2 scale-to-BF16 output
boundary, also checking its gate/up counterpart to avoid an incomplete claim.
`tools/proj_g_short_scale_feasibility.py` pins and extracts the actual helper0
literal from upstream G0 (raw SHA256 `1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a`,
owning job `3a5f879df0d44825`) and607e (`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`,
owning job `607eb52bdaf24160`). Assignment anchors are12583 and12481 respectively.
The machine evidence records all three selected expressions and literal hashes.

G0 already multiplies every gate/up/down accumulator by the effective global
scale.607e inserts the BF16 round after that product and before downstream
SiLU or output-normalization work:

```cpp
// G0 down output
v0f = accumulator * gsc * nw0;
// 607e down output
v0f = fork_bf16(accumulator * gsc) * nw0;
// Both sources form the same effective scale, before the output expression
gsc = global_scale * TWO62;
```

The changed contract is the explicit BF16 stage value, not a newly introduced
global-scale multiply. Its baseline/interface rationale remains in the pinned
`2026-11-01-g-upstream-precision-inventory.md` and
`2026-11-01-g-fork-precision-contract.md`; this screen does not demand bitwise
equality of arbitrary upstream FP32 intermediates. The downstream multiply must
consume the rounded stage value. Removing or moving that boundary changes it.

## Why the obvious scale representation is not exact

Official `muse_decode_layer/reference.py:100-104` produces per-expert FP32
global scales and multiplies them by `exp(randn)`. The interface does not
restrict them to powers of two. `TWO62` in the native implementation does not
make their product a general power-of-two scale.

Actual CPU Torch BF16 conversion gives this positive finite counterexample:

|Quantity|Value|
|---|---:|
|Accumulator|1.00390625|
|Effective scale|1.0078125|
|BF16(accumulator * scale), widened|1.015625|
|BF16(BF16(accumulator) * scale), widened|1.0078125|

Both inputs are exactly representable FP32. Thus moving the scale after the
round, even with a second round, cannot remove the dependency while preserving
the required value. Folding it across that round into normalization coefficients
has the same obstruction. Pre-scaling/repacking weights changes the MMA inputs
and accumulation path and would need a different numerical proof; it is not a
minimal exact restoration established by this source screen. No such candidate
was generated. Nor is replacing the product with an FMA-plus-zero a demonstrated
dependency reduction or a universal signed-zero-equivalent transformation.

## Actual instruction budget and stop

The existing selected Tiny/Tiny2 CUDA12.8 screen in `aafc32d` shows the scaled
product feeding F2F, widening, and SiLU/output math. Its paired intrinsic screen
already tested the available paired RNE representation once: added PRMT work
and an UP-to-gate dependency, with unchanged resources. This follow-up does not
reopen manual unpacking, gamma masks, scaling adapters or a compiler sweep.
There is no newly demonstrated instruction/dependency elimination here.

Holding the other four loads and baselines fixed, the owning607e short-load
times require an equal2.753184us reduction in both short loads for a1% whole-six
factor (about3us gives1.010916763). This is the prior arithmetic scenario, not
measured exposed boundary time or a speedup prediction. No source proof or
component measurement in this bounded screen supports such removal.

Status **CLOSED_NO_EXACT_DEPENDENCY_REMOVAL**, a source-feasibility closure,
not a GPU performance failure. The CPU evidence script completed successfully;
`evidence/2026-11-01-g-short-scale-feasibility/audit.json` contains both exact
source/helper pins, source expressions, reference hash and counterexample.
There is no timing harness, runtime test, resource/admission claim or score.

Stop lane3 now. The separately owned short final-counter RMW lane has an actual
isolated compiler delta and remains the priority after common-control progress
qualification; this report does not qualify or schedule it. Reopen this lane
only for a new exact representation with a demonstrably removed dependency,
preserved BF16 and ordering/progress contracts, and a defensible whole-six
budget. Lt guard restoration, unresolved DF/YBF/MFIN proofs and unrelated
projection tuning are not replacement candidates from this pass.
