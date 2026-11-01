# Standalone projection: dispatch, precision and next-reset gate

2026-11-01. Evidence checkout root `1b13bec`; owned audit branch `58f4da7`.
No GPU dispatch, external kernel import or optimization change.

**Decision: no new check justified by this comparison.** The four largest adjusted
latency gaps are also the four largest single-cell score opportunities, but three use
BF16-rounded split reductions. The remaining router opportunity overlaps already
measured changes and has a small residual ceiling. No new supported mechanism emerged.

## Exact owners and dispatch

Comparator `3c6a2baf7b4b4bf3`, H200, pass9,1.573015516x CI[1.549420283,1.596610748]:
keell0renz commit8e30e1be994db4310794b8d04e90d18d895deb84, standalone file raw SHA
922da8e54073b2836610d2832e06cf694dba691362dfc9969fc2e50c108c591b
(LF-normalized abdcdc0100da6ed97ac289b345f8c1358d054fdf58c45febcc9f9eceb1513c09).
Receipt payload7d12a581...; do not confuse server content hash with file SHA.
Owned C2 scored `9eba96894ee1402e`, H200, pass9,1.527806483x
CI[1.504889,1.550724], source3044ee7/comment repeat. Persistent root
`projection_best.py` is AST-identical to frozen C2 b2c4c096...; its current raw SHA
5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4.

| Cell | C2 actual path | Comparator actual path | Precision/descriptor finding |
|---|---|---|---|
|pooler128|Lt L1, algo66/tile15/stage35/custom4/cluster2, splitK1|chenwainuo rq d2s5hzwP, KS4|Comparator is **not** its ag Lt pin; FP32 WGMMA partials round BF16 before BF16 tensor reduce-add. C2 uses full-K FP32 compute/BF16 output.|
|QKV128|wd<1,128,3,1,false,8>, b1n128c3s8h|same rq variant, KS7|C2 uses raw BF16 weights and FP32 DSMEM reduction, one final BF16 conversion; comparator uses exact12-bit weights but BF16 partial reduction.|
|router256|wd<1,128,8,1,true,6>, r128c8s6k|vjeux rk,64x64,S8,6-stage24KiB ring|Both BF16 hi+lo weights/FP32 accumulation/output. Comparator uses global FP32 red.add after in-kernel clear; C2 clustered reduction. No Lt plan involved.|
|unpool128|Lt L3,algo66/tile18/stage35/custom2/cluster5,splitK1|same rq variant,KS2|Again BF16 partial reduction versus C2 FP32-compute Lt, not a missing known Lt pin.|

The final `_TABLE` fixes scored shapes; V6 nearby-shape tuning does not override them.
Prepare logs force d2s5hzwP for all three128 BF16 cells, and all three retained profiles
show `rq<128,5,1,0,0,0,42999939,1>`. Router profile shows `rk_kernel`.
[Pinned dispatcher](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_bf16_proj.py#L11323),
[rq dispatch](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_bf16_proj.py#L4806),
[BF16 packing/reduction/map](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_bf16_proj.py#L5394),
[rk implementation](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_bf16_proj.py#L9215).

Compiler audit: rq uses-O3,sm90a,DNDEBUG,relaxed-constexpr; rk and our wd use-O3/sm90a.
No fast-math flag in these selected kernels; comparator's fast-math at3137 is the flush
calibration extension. C2 Lt requests compute32F and BF16 operands/output; its two
selected pins are already validated, not newly discovered algorithm IDs. Neither
compiler flags nor lossless weight packing establish FP32-equivalent split reduction.
The public rq/rk also use software global zeroing joins under ordinary launches; a
future implementation needs an actual residency/ordering proof. A passing receipt
does not supply it. This audit does not infer a blanket rules violation from rounding;
it declines to treat that precision-changing mechanism as measured unchanged-precision gain.

## Official score sizing, all30 cells

All cases carry weight62. The official aggregate is **six load ratios after summing
five operator times per load**, then geometric mean. This reconstructs both headline
scores exactly; geometric mean of30 independent cell ratios does not.
Times below are receipt `median_ns`/1000 (paired-ratio-derived, replay-adjusted).
For each cell, delta=C2-comparator and single-cell factor is
`(C_m/(C_m-delta))^(1/6)`. Signed percentages are arithmetic counterfactuals, not paired
gains; negative values retain regressions rather than cherry-picking.

|Layer|M|C2 us|Comparator us|Delta us|Single-cell score change|
|---|---:|---:|---:|---:|---:|
|qkv_gate|1|9.058|9.268|-0.210|-0.0991%|
|o|1|5.036|5.002|+0.034|+0.0162%|
|pooler|1|13.516|13.393|+0.123|+0.0582%|
|unpooler|1|3.734|3.571|+0.163|+0.0775%|
|router|1|3.838|3.763|+0.075|+0.0357%|
|qkv_gate|8|9.771|9.060|+0.710|+0.3177%|
|o|8|5.476|5.350|+0.126|+0.0557%|
|pooler|8|13.419|13.945|-0.526|-0.2308%|
|unpooler|8|4.535|4.490|+0.045|+0.0200%|
|router|8|4.468|4.329|+0.138|+0.0613%|
|qkv_gate|16|9.741|9.945|-0.204|-0.0722%|
|o|16|5.366|5.634|-0.267|-0.0947%|
|pooler|16|13.800|13.948|-0.148|-0.0524%|
|unpooler|16|13.557|13.976|-0.419|-0.1480%|
|router|16|4.449|4.707|-0.258|-0.0913%|
|qkv_gate|32|10.557|10.479|+0.078|+0.0252%|
|o|32|5.869|6.035|-0.166|-0.0534%|
|pooler|32|15.084|15.076|+0.008|+0.0025%|
|unpooler|32|14.721|15.071|-0.351|-0.1125%|
|router|32|5.490|5.517|-0.026|-0.0085%|
|qkv_gate|128|14.664|12.118|+2.547|+0.6470%|
|o|128|6.812|7.014|-0.202|-0.0500%|
|pooler|128|19.994|17.187|+2.807|+0.7147%|
|unpooler|128|18.987|17.458|+1.529|+0.3848%|
|router|128|6.639|6.137|+0.501|+0.1251%|
|qkv_gate|256|16.941|17.038|-0.096|-0.0195%|
|o|256|8.338|8.765|-0.427|-0.0860%|
|pooler|256|24.434|24.456|-0.022|-0.0044%|
|unpooler|256|23.441|22.924|+0.517|+0.1050%|
|router|256|9.261|7.170|+2.091|+0.4293%|

C2 load sums for M1/8/16/32/128/256 are35.182617,37.668069,46.914076,51.720988,
67.095785,82.416369us. Four-cell joint substitution reduces C128 by6.881962us and
C256 by2.091484us. Holding every other cell and C2 baseline fixed:
`((67.095785/60.213824)*(82.416369/80.324885))^(1/6)=1.022571615`.
Thus **+2.2572% optimistic headroom**, C2-score equivalent1.562291543. This is neither
a measured improvement nor a ranking forecast. Different inputs, machines and replay
calibration across receipts prevent causal attribution or confidence intervals on this bound.

## Existing attempts and negative gate

- BF16128: rs128 itself is expressly closed for this task. FP32-compute alternatives
  already include broad Lt sweeps (047bc3a4168e42d6/d6eaf1e73db047a5/44fa08c6d2f44b44),
  FP32 split-K Lt8b9e8b7c1ecd4c3e, raw large-tile residency4ca25ee4eccc495c and phase
  ablations e6e5bd6d0a104faf, decoded WGMMA4922f366a0c142ca. QKV128 depth2
  4542b2b04cb344ec and owned push/transpose4edaa20c98904404 lose. These do not prove
  every new architecture fails; this comparator supplies no new legal FP32-reduction
  implementation that clears their reopening gates.
- Router256 full observed gap alone is only **+0.4293%**. Order/join/stage/barrier/box
  attribution3fd0c7a7cea74ab4/e9d9499a465d4548 and SASS-driven descriptor/prefetch
  3170985420f74869/7d2fe593fdfa4720 already cover the named differences. C3's measured
  .62–.69us component gain already addresses part; subtracting it for sizing leaves
  ~.286–.301% score ceiling. Fresh C3 paired measurements would be needed to establish
  the residual. Do not repeat join/cap/stage/FV sweeps without new phase/SASS evidence.
- The next low-token gap is QKV8: .710us, **+.3177%**, fifth overall. Its comparator
  is register-direct `gv<1,1,1,2,16,8,1,1,2,1>`, not the older Nathen sk2 route.
  Register-direct qkv8 already lost+1.70/+2.24us in1759290da58047ab/bcabe48caa884e67.
  Current profiles also show gv8.9us versus owned sk7.6us, so profile durations do not
  establish a causal gain corresponding to adjusted medians. No fresh mechanism gate.
- Per-us leverage is higher at low M, but available gaps remain small: slice/unpool1
  .163us→+.0775%, router8 .138us→+.0613%; existing cgemv layout/register/escape and rt
  experiments cover those families. Unpool256's larger .517us is worth only+.1050%.

**Next action: preserve C2/C3; spend no next-reset check on these knobs.** Reopen only
with a distinct precision-preserving source/schedule and component evidence satisfying
the scoped closures, a score budget large enough for the stated objective, then one
bounded same-process owned A/B with duplicate controls, FP64/probes/fresh inputs and
actual resources. Never import the external standalone code. New measurement, if later
justified, must beat2x duplicate log noise on full official scoring and retain all fences.
