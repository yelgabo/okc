# OFF common pre-expert coefficient parity repair

2026-11-01. Separate inherited-interface work authorized after the dense
inventory `87649fd`. No integration cost attribution, performance claim,
production edit, GPU/service/official call or broad harness. Root owns index
and integration. Existing607e and private artifacts remain unchanged/HOLD.

**A bounded source-only repair is feasible:** all raw pre-expert coefficient
reads are direct arguments to seven Triton consumers. The OFF builder preserves
the raw parameter and field, prepares a distinct BF16 effective coefficient once,
and updates all consumers. Three CPU source/coefficient tests pass. This is
not a newly qualified whole-layer candidate.

## Baseline path actually verified

`muse_decode_layer/baseline.py::_norm` installs `_param(w)` without changing
its dtype, calls `precompute_effective_weight` and returns that module.
Baseline222 selects it as `pre_expert_norm`, then337 calls the module on
the BF16 pooler output. Weights' declared pre_expert_norm dtype is BF16
(`submission.py:97`); reference97 generates it as BF16 as well.

Pinned [MuseSpark1xRMSNorm81-110](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/muse_spark_layers.py#L81)
stores `(self.weight + self.gain_center).detach()` and passes this effective
weight to `F.rms_norm`, returning activation dtype. With this BF16 parameter
and gain_center1, the stored coefficient is BF161+w. This is a concrete
coefficient interface, not a claim that every internal FP32 arithmetic step
must match a baseline bitwise. Primary source SHA is already pinned by the
precision-contract ledger.

G0 and607e both store raw `g_preexp` and compute1+raw.float in route kernels.
This difference is inherited, not a repair introduced between the two measured
files. There is no measured performance cost or evidence that it explains their
headline gap. No categorical contest eligibility verdict follows from merely
observing this coefficient difference; the purpose is explicit baseline parity.

## Exact OFF builder and enabled source

`tools/proj_g_preexpert_gamma.py::build(source,enabled=False)` requires exact
607e SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
Disabled output is byte-identical to that869076B input. Opt-in output:

- 869292B; SHA256 `4b3b42632d43cf95e830491993df66b1117fd7e56200b5d3b113ee45b98729a3`.
- Persistent scratch `.build/today-proj/preexpert-gamma/on.py`; manifest copied
 to `evidence/2026-11-01-g-preexpert-gamma-off/manifest.json`.

No normal file is overwritten. There is no runtime global flag, shape/time
selector or extra timing/calibration. The only preparation addition is:

```python
st["_g_preexp_effective"] = (st["g_preexp"].float() + 1.0).to(torch.bfloat16).contiguous()
```

Original raw `st["g_preexp"]` and `weights["pre_expert_norm"]` remain intact.
The new BF16 pointer has the same4096-element consumer ABI; all15 direct
kernel arguments now pass it. The eight `1+tl.load(GPRE...)` expressions
become `tl.load(GPRE...)` in these seven functions:

| Consumer | Loads | Why retained |
|---|---:|---|
|`_route_b`|1|two-pass fallback|
|`_route`|1|generic partial-sum fallback|
|`_route_split`|1|split pooler/router fallback|
|`_graft_route`|1|large/ordinary graft route|
|`_sl_graft_route`|1|short graft alternative|
|`_sl_tile_route`|2|selected short PRELOAD true/false branches|
|`_gle_route`|1|currently disabled deferred alternative; keep ABI consistent|

The census rejects unknown GPRE occurrences, unknown/non-direct field users,
changed load counts or a non-pinned source. No consumer is left adding1 to
the already effective coefficient. Selected/fallback routing topology,
normalization denominator/epsilon, BF16 output rounding, fences/barriers,
expert ABI and all embedded native CUDA/CPP literals remain otherwise unchanged.
The existing tuner remains present; unchanged source does not guarantee identical
future tuning/routing outcomes after a numerical correction.

## Actual CPU gates and limits

`tools/test_proj_g_preexpert_gamma.py`: **3 tests pass in3.484s** using the
existing CPU Torch environment. They cover default-OFF identity/source-pin
rejection; exact generated-source reversal; an independent AST transform
of only the eight adds,15 call arguments and preparation assignment; and
execution of the actual generated coefficient assignment.

The executed assignment matches literal baseline `(raw_BF16 + 1.0).detach()`
bit-for-bit for **65,280 finite BF16 bit patterns**. Raw bytes/object identity
are retained; derived coefficient storage is distinct and contiguous. A
half-ULP example confirms the inherited FP321+w and BF161+w representations
can differ. This is coefficient validation, not full norm or layer numerical
qualification. Nonfinite model arithmetic, CUDA/Triton lowering, FP8 caches,
graph captures, progress and H200 performance were not executed.

## Composition and stopping rule

This exact-pin builder is **not directly composable by calling it on an already
modified source**; it deliberately rejects the other epoch. A coordinator must
define a separately pinned combined transform and verify disjoint deltas before
combining it with the OFF ATT7 cooperative host repair or short-counter treatment.
All such combined sources still require the common native/attention progress
proofs and actual mathematical/ownership/capture validation. A private owner
clone must account for the additional immutable coefficient tensor as well.

Keep this correction common to both comparison arms if used. Do not present it
as a restoration to G0 speed, a measured optimization, or permission to remove
other BF16 boundaries. Stop on an incomplete consumer census, double1+effective,
mutated raw weights, changed fallback ABI or unresolved common progress. No
submission or GPU qualification is authorized by this source milestone.
