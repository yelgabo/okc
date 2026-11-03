# G0 to 607e: minimal dense integration inventory

2026-11-01. Scoped human priority: source/CPU audit only while the wider goal
is paused. Reconciled requested main `0f51f77` with newer root files, reading
the integration handoff, precision contract, original projection audit and
ruled-out entries first. Exact source hashes below remain unchanged. Root owns
README/integration. No GPU, official/service call, tuning, source arm or harness.

**No selected short dense geometry or saved Lt plan changed.** The major
dense dispatch change is the removal of rs128 on three large-M routes. The
ordinary compressed kernel's math is unchanged; required downstream BF16
rounds and disabled deferred boundary rewrites are distinct from conservative
internal-reduction restrictions. A safely conforming whole-layer arm cannot
currently be generated merely by turning off the conservative restrictions:
inherited progress and coefficient-interface dependencies remain.

## Pins and reproducible machine inventory

- G0 measured file: keell0renz commit `8e30e1be994db4310794b8d04e90d18d895deb84`,
  raw995999B/SHA256 `1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a`,
  root `.build/today/kel/kernels/muse_decode_layer.py`; owning3a5f879df0d44825.
- 607e: `muse_decode_layer/cand/g1b4_stage_release.py`,869076B/SHA256
  `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`,
  owning607eb52bdaf24160. Later OFF/packed candidates are not substituted.

`tools/proj_g_dense_delta_inventory.py` validates both hashes, all12 matching
load/context/shape/weight cells, unchanged config bindings and unchanged ASTs
for final `dense`, `_graft_layouts`, `_graft_lt_run`, `_pool_router_fp32`,
`bfp__sk2_geom` and `bfp__sk2_src`. Output:
`evidence/2026-11-01-g-dense-minimal-delta/inventory.json`. It retains each
cell's actual profiler signatures/calls, source configs, Lt descriptors,
contracts, flags and arm feasibility. It deliberately does not assert equal
job seeds/input banks. Profile samples are existing observations, not costs
attributed to a repair. No speedup forecast is made.

The SK2 config list, picks, K dimensions, compression set, seven opaque Lt
saved word arrays, QKV256 attribute-built plan and Lt target token set are
identical. Equal function ASTs do not imply equal whole-pipeline behavior:
global PDL, `_rq_gemm`, boundary functions and ownership changed separately.

## Selected projection path in every retained cell

Each table entry applies to the named exact receipt; future packing failure
or self-test rejection can take a preserved fallback. Full signatures/config
arrays and both profiles for all four operators are in the machine inventory.

|Cell|QKV, G0 to607e|O|Pooler/router|Unpooler|
|---|---|---|---|---|
|d1 sliding|qkv8_r176 SK2, same|GLE O+deferred boundary to ordinary o8h_r88 + boundary|GLE_FRT `pooler`, retained|slice8_r192,1024 rows, same|
|d1 full|same d1 selection|same d1 change|same retained GLE_FRT|same slice|
|d16 sliding|qkv16_r176, same|o8h_r88 bucket16, same|SK2F `pooler`, same|unpooler16h_r100, same|
|d16 full|same d16 selection|same|same|same|
|d32 sliding|qkv32_r208, same|o32h_r100, same|SK2F `pooler`, same|unpooler32_r144, same|
|d32 full|same d32 selection|same|same|same|
|v1 sliding, T8|narrow qkv8 override,5 stages, same|ordinary o8h_r88, same|GLE_FRT `pooler`, retained|slice8_r192,1024 rows, same|
|v1 full, T8|same narrow override|same|same|same|
|v16 sliding, T128|rq d2s5hzwP to existing saved Lt|same saved Lt|same FP32-output Lt pr plan|rq d2s5hzwP to existing saved Lt|
|v16 full, T128|same T128 change|same|same|same T128 change|
|v32 sliding, T256|rq n256s5m1hzmEH to existing attribute-built Lt|same saved Lt|same FP32-output Lt pr plan|same saved Lt|
|v32 full, T256|same T256 change|same|same|same|

Important non-dormant details:

- Short PR does **not** use static `pooler8h_r72`/`pooler32_r144` picks when
  the active fused path is selected. GLE_FRT and SK2F explicitly derive the
  `pooler` configuration. The profiler's NT4/WSK8 signature corroborates it.
- T8 QKV is the existing `_de_source` narrow extension with5 stages. Blindly
  instantiating static qkv8's6 stages would change the arm. Both receipts
  already select the narrow route; it is not an integration delta.
- AtT128 G0's rq profile aggregates **two calls** (QKV and unpool);607e's
  same-name Lt row aggregates QKV and O. Individual times cannot be recovered
  by dividing or assigning those aggregates. All eight Lt plans are reported
  accepted in607e startup; the splitK guard causes no observed missing plan.
- `_rq_gemm` is consulted before Lt only for an existing plan; upstream's
  `chen_rq` contains QKV128/256 and un128, confirmed by its owning bake log.
  Other baked/dead variants do not describe active dispatch.

## Concrete numerical/host contracts and minimal deltas

**Operands and projection outputs.** Baseline274 performs QKV BF16 GEMM before
Q/K norm and V/gate slicing;310 BF16-rounds attention value before output norm;
314 projects BF16 O;317 returns BF16 pre-FFN pooler input plus a distinct short
FP32 router vector.322 forms BF16 pooler output before pre-expert norm.340-349
consume BF16 expert/unpool results and the final boundary. A fused producer
may avoid materializing global tensors, but must preserve those stage values.
This is not a rule requiring arbitrary internal values/reduction orders to be
bitwise identical. Baseline/reference identities are pinned by the existing
607e interface proof.

**Ordinary SK2.** Upstream/607e use lossless BF16 weight packing, FP32 MMA and
FP32 escape corrections, one BF16 projection result. The ordinary CUDA literal
delta is exactly two conditional async global proxy fences and host PDL
attribute1 to0, as independently proven in the preceding precision inventory.
No tile/stage/weight-codec/math rewrite occurred.607e setsPDL=False;
`_bfp_pdl`10122 sets the per-call mode0, so added `if(pdl)` fences are inactive
under that policy. Losing overlap and executing extra fences are different
questions. Re-enabling PDL needs the common dependency/ordering proof; source
AST equality cannot supply it.

**Pooler/router.** ForT<=8, pooler h remains BF16 while router hr is FP32;
`_pool_router_fp32`16642 retains GLE_FRT on those two inputs. ForT>8 the
concatenated BF16 pooler/router-hi/router-lo weight produces4608 FP32 values.
Pooler4096 values round through BF16 before their RMS; router adds two FP32
hi/lo dot products. Keep `_sl_tile_route`15794 and large `_graft_route`12069
casts. Feeding hr to pooler or BF16-upcasting h for short router changes the
source interface. GLE_T's two-row reconstruction moves nonlinear BF16 rounds;
B1X similarly changes the branch round before residual. Their disablement
is not interchangeable with disabling GLE_FRT, which remains active.

**Lt.** `_GraftLt` requests compute32F (68); input layouts are BF16. PR output
is FP32, QKV/O/un output BF16. Feature batching divides N, not K. Saved words
and `_graft_layouts` are identical; QKV256 remains id66/tile124/stage35/
splitK1/reduction0/custom0/cluster2.607e additionally queries splitK and accepts
only0/1. This is conservative: a specifically proved FP32 partial/reducer plan
could be compliant, and splitK alone does not prove reduced precision. Removing
the guard without introducing a new verified plan changes none of these eight
accepted routes. No new plan search is part of this audit.

**rs128.** It retains BF16 operands and FP32 WGMMA accumulators, but `rq_pk`
15187 and15688-15695 round each K partial to BF16 before BF16 TMA reduce-add.
The existing precision-contract report correctly classifies its exclusion as
an unresolved conservative internal-reduction policy, not a categorical
all-partials-FP32 rule. However restoring this path is not just a precision
decision: the ZI path15415-15438 has each KS CTA zero part of an output tile,
release-increment a GPU counter and acquire-wait for all KS peers, then publish
shared zflag. Consumers15679/15763 wait for it. All waits have fatal timeout
traps; there is **no silent fall-through finding** here. `rq_go`15846 nevertheless
uses ordinary launch. Static one-wave sizing does not itself request cooperative
admission. A per-tile peer-progress/admission proof/repair is still needed.
This is an inherited eligibility dependency, not an observed hang, measured
regression or new reason attributed to the607e score gap. No RQ variant built.

**Compiler/host policy.** Ordinary bfp is O3/nativeH200 sm_90a without fast-math;
GLE_FRT is O3/sm_90 without fast-math; upstream RQ adds NDEBUG and relaxed
constexpr to O3/sm_90a, not fast-math. Original PDL=True becomesFalse and several
explicit launch attributes change independently of the numeric repair. Existing
state-specific attention scratch, immutable per-call prefetch, thread-local
GLE host argument and Lt workspace ownership must remain in any common arm.
Dormant wide/GQ/fallback functions and unselected diagnostic flags are not
evidence that these selected configurations changed.

## Shared effective-gamma interface dependency

Actual baseline `_norm`149-154 installs the original BF16 parameter and calls
`precompute_effective_weight`; baseline222 creates this module for pre-expert
norm. Pinned [MuseSpark1xRMSNorm88-110](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/muse_spark_layers.py#L88)
caches `(self.weight + self.gain_center)` in parameter dtype, and `forward`
passes that effective weight to `F.rms_norm`. Thus this baseline path consumes
a BF16 effective1+w coefficient, then returns BF16 activation.

Both G0 and607e instead retain raw `g_preexp` (607e11148) and active route
functions compute1+raw.float in FP32 (for example15804). Graft boundary
effective weights are already explicitly BF16 (12462-12464), so they are a
different ABI. The pre-expert difference is a shared **interface parity
dependency**, not a newly introduced integration delta or a universal demand
for bitwise equality of all arithmetic intermediates. No cost is charged and
no categorical contest-rule verdict is inferred. Any common repair must keep
the raw weight immutable, prepare the derived coefficient once and update all
selected consumers consistently; a partial rename would multiply by1+effective.

## Can existing tools safely emit the proposed minimal arm?

**Not as a ready conforming arm with a one-call option.** Existing
`dl_fork_projection.patch_projection` unconditionally couples RQ tail removal,
two proxy fences and Lt guard. `dl_build_attributed_fork` also installs stage
casts, GLE disables and global PDL/ownership repairs. Merely omitting those
transforms reintroduces unrelated defects; selectively retaining RQ requires
its precision/progress decision and prepare/packing/counter-state dependencies.
Its calibration/MFAB wrappers must not silently alter the common expert policy.

Two sharply defined source plans remain:

1. Keep RQ excluded, accepted Lt plans unchanged, mandatory stage rounds and
   ordinary upstream SK2 arithmetic. This is a feasible component extraction
   plan, but already matches607e's dense policy; it supplies no distinct dense
   treatment. Common native/attention progress and gamma dependencies still
   need a separate source epoch and qualification before a whole file is called
   conforming. The newly identified inherited ATT7 ordinary-join issue is not
   charged as an integration cost.
2. Retain upstream RQ or a different splitK plan. This is a genuinely different
   internal-reduction arm, not the above minimum. It needs an explicit allowed
   precision contract, verified progress/admission/publication, exact packing
   and state ownership, fallback semantics, byte/prepare accounting and numerical
   qualification. None is bypassed by the old public pass or source availability.

Stop at this inventory until those separate decisions are assigned. The JSON
marks `generated=false`, `conforming_claim=false` and forecast null. No source
arm, compiler sweep, full benchmark harness or performance prediction is emitted.
