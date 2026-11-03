# Layer large-M projections: measured stock alternatives and negative gate

November 1. Root evidence checkout `37c261cf95e8acfe5a9cfb5897a16ad67c22b1c`;
owned branch parent `a9ca837`. No GPU, main-checkout changes or candidate changes.

**Decision: no additional G1b3 host dispatch is justified from these receipts.**
G1b2 already uses eight FP32-compute, nonsplit cuBLASLt plans. Its larger dense
deficits versus G0 correspond mainly to the disabled BF16 partial-reduction rq
paths. Replacing every positive large-M dense row gap with the G0 duration gives
only **+0.42167%** score arithmetic; that is not a legal replacement mechanism or
an exposed critical-path measurement. The strongest owned alternative has only
**+0.33991%** optimistic group arithmetic, without isolated per-op attribution.

## Frozen evidence and boundaries

- G0 `3a5f879df0d44825`: exact keell0renz layer at upstream `8e30e1be`,
  pass13, score2.248790532, CI[2.215058674,2.282522390].
- G1b2 `736a5cbc977f4ad4`: pure file SHA256
  `258c848c662149f9c6ac178039f179de6f0b0ec4cb0b503451d2974c7a7a8dfc`,
  pass13, score2.065558557, CI[2.034575178,2.096541935]. Native expert repairs,
  precision/order changes and MFIN/DF/YBF/PDL-off policy are bundled differences.
- G1a `606f47f251f34873`: owned expert fallback with the same graft projection
  family, pass13,2.010975134. Dense rows do not improve systematically over G1b2.
- Owned clean Lt+short-boundary `47bec92021ef4bbe`, pure SHA256
  `851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a`,
  pass13,2.010247860. Its source and prior paired checks establish the stock
  precision/ABI path; cross-receipt profile rows do not establish a port speedup.

Receipts and human-readable profiles are under `muse_decode_layer/profiles/`:
`muse_decode_layer.3a5f879df0d44825.{json,txt}`, `g1b2.profile.736a5cbc977f4ad4.*`,
`g1a.profile.606f47f251f34873.*`, `combo_pure.47bec92021ef4bbe.profile.*`.
Independent analysis tool `tools/dl_proj_budget_audit.py <root-checkout>` verifies
that official weighted aggregation reconstructs all four headline scores exactly.

## Actual dispatch and legal stock alternatives

G1b2 startup log line3 confirms all eight plans selected: qkv/O/pr/un at128/256.
The submitted source `muse_decode_layer/cand/g1b2_native_experts.py` has:

- `_GraftLt`12140: compute type68 (`CUBLAS_COMPUTE_32F`), float alpha/beta,
  BF16 weights/inputs, BF16 outputs except **pr FP32 output**. `pr` is pooler plus
  router hi/lo rows, prepared by concatenation at11148; a pooler-only standalone
  result does not substitute for this operation or its FP32 boundary.
- `_GRAFT_LT_PICKS`12206 and `_graft_layouts`12193: qkv128 TN/output-feature
  batch16; O128 NN/batch8; un128 NN/batch2; pr128 TN/batch1. At256, O/un NN/batch2,
  pr TN/batch2. Output-feature batching partitions N and still computes the full K.
- `_GRAFT_QKV256`12215: algo66/tile124/stage35/custom0/cluster2/splitK1,
  **the same owned standalone pin already ported and measured in the layer**.
  `_fork_lt_no_split`12222 checks opaque plan splitK0/1 before accepting it.
- `dense`12295 and attention entry12438 select these plans; `_rq_gemm`16748
  returns None after the conservative repair. G0 instead overrides qkv+un at128
  and qkv256 with rq, whose BF16-rounded split partials are not an unchanged-FP32
  reduction alternative. This is a precision distinction, not a blanket rules verdict.

The owned `tools/dl_proj_lt_port.py` pins un128 L3 algo66/tile18/stage35/custom2/
cluster5; un256 L1 algo66/tile20/stage35/custom4/cluster3; qkv256 as above. L3 is a
row-major transposed-weight descriptor; L1 is a column-major transposed-weight
descriptor, both full-K BF16 operands with FP32 compute. They are not the graft's
strided output-feature batches. All three passed full-FP64, one-hot and fresh-graph
gates in `1e3349e377234e89` and the clean combined path. That paired check improved
**the older owned fallback**, not G1b2: verify16 +0.49%, verify32 +1.29%, two-load
aggregate +0.89%. It does not justify reapplying the same algorithm ID as a new win.

## Twelve cells and official score budget

Each load is `(46*sliding + 16*full)/62`, then geometric mean of six load ratios.
Whole-cell adjusted medians below are separate from the single retained profiler
sample's dense-row sum. Dense sums include the selected projections; they are not
stage-exclusive durations. GLE includes boundary work at small M. Overlap/PDL waits,
different seeds, expert scheduling and cross-run conditions confound causal savings.
No activity union is called SM utilization, and no row subtraction is called idle time.

|Load|Context|Weight|G1b2 whole us|G0 whole us|G1b2 dense-row us|G0 dense-row us|
|---|---|---:|---:|---:|---:|---:|
|decode1|sliding|46|76.923|66.352|40.3|39.8|
|decode1|full|16|83.615|73.527|38.9|39.5|
|decode16|sliding|46|132.798|130.187|49.4|50.8|
|decode16|full|16|212.540|205.802|49.3|51.9|
|decode32|sliding|46|159.381|152.172|54.6|58.3|
|decode32|full|16|310.008|301.541|51.6|61.9|
|verify1|sliding|46|110.857|100.606|38.5|40.4|
|verify1|full|16|114.062|112.279|38.5|38.6|
|verify16|sliding|46|266.664|237.373|69.7|64.4|
|verify16|full|16|350.368|319.902|70.5|62.5|
|verify32|sliding|46|336.668|308.691|86.8|85.9|
|verify32|full|16|499.824|471.851|88.2|84.6|

G1b2 weighted load times are78.649701/153.376242/198.252333/111.684242/
288.264897/378.772522us in the table's six-load order. Only the last two loads have
the large-M dispatch under consideration; the other four remain fixed in sizing.
Positive dense gaps5.3/8.0/.9/3.6us yield weighted reductions5.996774 and1.596774us.
The optimistic factor is
`((288.264897/282.268123)*(378.772522/377.175748))^(1/6)=1.004216698`.
Its G1b2-score equivalent is2.074268393, versus an8.87082% headline score ratio
between G0 and G1b2 (which also includes reference/calibration differences).
This is a cross-receipt sizing envelope, not a hard hardware bound or forecast.

The stock legal comparison is narrower:

|M/context|G1b2 QKV+O us|Owned QKV+O us|G1b2 un us|Owned Lt un us|G1b2 pr us|Owned concat us|
|---|---:|---:|---:|---:|---:|---:|
|128 sliding|28.1 (2calls)|14.8+7.8|16.6|21.2|25.0|27.4|
|128 full|28.8 (2calls)|14.5+7.6|16.4|21.7|25.3|25.9|

The owned QKV128 path is raw-BF16 WD `wd<1,128,3,1,false,8>` with FP32 DSMEM
reduction and final BF16 conversion (`layer_attn31s.py`16557,20975). Its O is the
owned fallback GEMM. G1b2's retained NNT kernel row combines QKV and O, so neither
28.1 nor28.8 is an individual QKV duration. Replacing the **combined group** with
the owned profile sums saves5.5/6.7us, weighted5.809677us; holding the other five
loads fixed gives1.003399066, or **+.33991%** / score-equivalent2.072579527.
No evidence separates an individual QKV port gain or proves its capture/launch
overhead in G1b3. The owned WD CUDA literal alone is41179 bytes; adding its packing/
host integration is not a slim Lt-only dispatch to a file already near1MB.

At256, owned and graft qkv use the same exact pin and both profile20.1–20.3us.
Owned unpool profiles26.2/26.6 versus graft25.7/27.0; owned O9.2/9.1 versus
graft9.4/9.2; owned concat32.5/33.6 versus graft31.6/31.7. No material, repeatable
stock advantage appears. Un128 is clearly slower in these owned Lt profiles.

## Stop and reopening criteria

Keep G1b3's projection dispatch unchanged. Spend the next layer checks on the
already registered conformance/critical-path question rather than these stock pins.
Prior layer front closures remain applicable: split router38497c81e7274f31,
pooler-vs-concat8bea43d605c74546 and one-CTA ring4098aa2361554229; see
`docs/ruled-out.md`269–270. Prior owned QKV128 depth4542b2b04cb344ec and
push4edaa20c98904404 also lost; neither is reopened by these profiles.

Reopen only with per-op same-input timing that isolates QKV128/O128 and an exposed
whole-layer budget, a compact exact implementation with stated source/prepare cost,
and a mechanism absent from those losses. Require two independent controls, common
prepared policy, both context/load seeds, >2x duplicate log noise on official scoring,
full-FP64/one-hot/fresh inputs and boundary ABI checks. No precision or fence changes
are authorized. No new dispatch generator or GPU check was produced by this audit.
