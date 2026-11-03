# G0 to 607e: precision inventory and the unexplained short-load gap

2026-11-01. Reviewer base `a20ec47`; root's newer precision-contract report
read from its checkout. No GPU/service calls, production changes, implementation
or quota request. Ordering/progress and native-expert audits are owned separately
by the other reviewers; this report does not duplicate or override them.

**The 5.77155% headline difference is real cross-run evidence, not an attributed
price for precision repairs.** Actual source changes explain why copying the
public file is not the same as reproducing its performance with the required
interfaces. They do not establish how many microseconds each repair costs.
No safe dense/router/attention precision rollback with a demonstrated material
whole-score budget emerged. The largest unresolved short-load lead is expert/
finalization behavior, not a missing dense codec or an assumed large-MFIN bound.

## Exact owners and arithmetic

G0: `3a5f879df0d44825`, PASS13, **2.2487905322335195**; exact public layer at
keell0renz commit `8e30e1be994db4310794b8d04e90d18d895deb84`,995999 bytes,
SHA256 `1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a`.
Its normalized profile names `.build/today/kel/kernels/muse_decode_layer.py`
as the submitted source. Owning receipt is `profiles/kel.3a5f879df0d44825.json`.

607e: `607eb52bdaf24160`, PASS13, **2.1260826764231098**; submitted
`muse_decode_layer/cand/g1b4_stage_release.py`,869076 bytes, SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
Owning receipt is `profiles/scored.607eb52bdaf24160.json`. Do not substitute
later stage-retire/production-packaged files when interpreting these timings.

`tools/proj_g_precision_inventory.py` verifies both source hashes, actual
literal deltas and all12 cells; evidence is
`evidence/2026-11-01-g-upstream-precision/inventory.json`. It reconstructs
both scores exactly. G0 evaluated with607e's baseline times would score
2.264796221982, a **1.065243721** ratio over607e; the actual score ratio is
**1.057715468** because each job has different baseline measurements. This
normalization is arithmetic, not a same-process paired measurement.

| Weighted load | G0 us |607e us|
|---|---:|---:|
|decode1|68.203874|78.549045|
|decode16|149.700710|158.241010|
|decode32|190.719032|191.739736|
|verify1|103.618625|115.990823|
|verify16|258.670706|269.347029|
|verify32|350.796973|359.297216|

Each load is46 sliding +16 full, divided by62; score is the geometric mean
of six baseline/entrant load ratios. The short loads have substantial retained
differences. No large-M-only bound closes them. Different weights/routing,
calibration and overlapping profiler samples confound component subtraction.

## Upstream to actual607e inventory

Line anchors below refer to the exact submitted files, not generic labels or
only the first fork transform. Pinned baseline/reference identities match the
607e evaluator image, per the existing
`2026-11-01-g-fork-precision-contract.md` interface proof and primary-source ledger.

| Change and actual scope | Requirement versus conservative choice | Existing cost evidence and priority |
|---|---|---|
| **Native attention decoded-value round:** upstream `pr8(x)` is identity;607e416-418 converts FP32 to BF16 RNE and widens. Calls at938,1099-1100,2847,2905,3023 precede value RMS and gate. No Q/K score/probability conversion is changed by this helper. | Strong baseline-stage requirement: baseline310 calls `o_norm_gate(out.to(BF16),gate)`. The literal judge reference uses FP32 attention for its tolerance oracle; that does not erase the baseline BF16 interface. Keep. | No isolated paired cost. Positive attention profile-gap scenario is only+0.10015% full score across12 cells; short-only+0.06934%. These are comparator sensitivities, not universal attention ceilings. Low restoration priority; deleting the round is unsafe. |
| **Fallback QKV boundary:**607e10394-10399 rounds reconstructed qe/qo/ke/ko/v/g before norm/cache/gate. | Required BF16 projection edge: baseline274 forms BF16 `_gemm` output before qk_norm and V/gate slices. Keep, including cases with FP32 split partial inputs. | No isolated paired cost; official retained profiles select native ATT7, not this fallback. Do not blame the active short gap on dead fallback conversions. |
| **Fallback E4M3 conversion and scaled-verify routing:** exact RNE helper replaces saturation/subnormal approximation; unqualified scaled verify delegates FA3, whose attention value is BF16-rounded before norm. | Preserve allowed cache/query dtype/quantization and stage ABI; not a tolerance-based speed switch. Required unsupported-route handling remains. | No isolated cost and no official12 scaled fallback selected in these profiles. Changing it cannot be credited with a measured official12 gain. |
| **Ordinary compressed SK2:**607e adds only two conditional global proxy fences after dependency waits; host PDL launch attribute changes1 to0. FP32 MMA/correction arithmetic and weight decoding otherwise identical. | Fences are conservative explicit proxy-ordering repairs, separate from precision. **607e PDL=False**, `_bfp_pdl`10122 sets `bfp_SK2_PDL=0`; these new `if(pdl)` fences do not dynamically execute under that policy. | No per-repair paired cost. Lost PDL overlap is a different ordering/scheduling hypothesis, owned by the ordering review. Do not attribute direct executing-fence overhead to607e. |
| **rs128 disabled:** upstream RQ_USE16209 selects qkv128/256 and unpool128. FP32 WGMMA partials are packed BF16 before BF16 TMA reduce-add;607e `_rq_gemm` returnsNone and uses retained Lt plans. | Conservative internal-reduction policy under unresolved rule scope, **not** a proven categorical ban on every BF16 split partial. It is not unchanged full-FP32 reduction. No permission inferred from tolerance passes. | No paired ablation. Positive large-dense group comparison+0.48455% six-load scenario; qkv/un128 share upstream rq aggregate rows, so no independent attribution. No short-load explanation. |
| **Opaque Lt splitK guard:**607e12185-12283 queries splitK0/1, full-K BF16 operands/compute32F, BF16 QKV/O/un outputs, FP32 pr output. | Broader-than-necessary conservative guard: a specifically verified FP32 partial/reduction plan could be safe. SplitK alone is not reduced precision. | Actual607e startup installs **all eight** QKV/O/pr/un128/256 plans. No rejected-plan fallback cost is observed. Removing the guard has no demonstrated benefit here. |
| **Projection/boundary rounds:** `_boundary1`10707 rounds reconstructed O before post-norm and normalized branch before residual; `_boundary2`10838 preserves unpool and normalized-branch rounds; `_graft_row_boundary{,_pre}`12020/12044 round reconstructed branch input. | Derived BF16 operator interfaces in baseline314-318 and final boundary348-349; fusion may remove global storage but cannot commute these rounds through later norms/linear maps. Keep. | No per-round paired timing. The final glue paths differ and some patched fallback functions are dormant. No count of cast statements is a runtime cost model. |
| **Pooler output before pre-expert norm:** route variants cast reconstructed pooler values through BF16, including actual `_sl_tile_route`15794 (else branch and output slices). | Baseline322 stores `_gemm(ffn_input,pooler)` in BF16 and passes that value to pre-expert norm; keep. FP32 router logits remain distinct. | No isolated timing. Current low-token total dense groups are already faster than G0. Repeating codec/tile experiments does not isolate these few casts. |
| **Deferred GLE_T and B1X disabled:** upstream two-row linear rewrite reconstructs normalized values after projection; B1X changes the post-norm branch round before residual, not just final router-vector dtype.607e GLE_T={}16216; B1X wrapper removed. | Required BF16-edge preservation plus independent ordering/ownership issues. Do not restore by fixing only a pointer or wait. | Bundled extra routing/boundary topology can matter, but no per-repair causal cost. Correcting the full rewrite is not a minimal precision restoration and must preserve nonlinear rounding. See ordering review for the separate proof. |
| **Short FP32 router preserved:**607e `_boundary1` stores unrounded FP32 hh to ROUTER_OUT; GLE_FRT=True and `_pool_router_fp32`16642 uses actual h/ hr separately. | Pinned JIT SGLang norm boundary supplies unrounded FP32 router values at T<=8 while the pooler branch is BF16. A fallback's BF16-upcast router is not the universal active contract. | No blanket extra BF16 router repair or missed FP32 router port exists. GLE_FRT is retained, not disabled alongside GLE_T. |

Transformation pins are reproducibly linked by `dl_fork_projection.py`,
`dl_fork_attention.py`, `dl_build_attributed_fork.py` and
`dl_build_g1_qualification.py::round_native_attention`. The first attention
transform alone leaves native CUDA untouched; the **later qualification tool**
changes pr8. An audit that stops at the first transform misses an active delta.
The evidence tool independently proves full ATT7 delta is exactly pr8 plus
the host PDL attribute, and ordinary SK2 delta exactly two fences plus that
attribute. ATT7 CPP is identical. This is stronger than trusting comments.

## What the short profiles support

These are retained profiler samples, not whole-operator critical paths:

|Cell|G0 dense / attention / expert / glue us|607e dense / attention / expert / glue us|
|---|---|---|
|d1 sliding|39.8 /5.2 /23.6 /4.5|38.8 /5.4 /29.8 /7.7|
|d1 full|39.5 /11.8 /21.8 /4.4|38.9 /12.1 /42.8 /7.2|
|v1 sliding|40.4 /7.2 /44.2 /7.1|38.5 /7.4 /71.6 /6.6|
|v1 full|38.6 /15.0 /57.6 /7.4|38.5 /15.0 /58.8 /7.1|

Replacing only positive short expert-row differences in607e's full six-load
arithmetic gives1.056961678; short glue gives1.006726342, attention1.000693403,
dense-group1.0. These factors are **not additive, causal speedups or attainable
savings**. They use different job banks and summed profiler categories; expert
job distribution and overlap may differ. They establish prioritization scope,
not why the same public code ran faster. Full12 evidence is retained; selecting
only the favorable short profiler rows for promotion is prohibited.

## Recommendation and next proof budget

The largest credible investigation is a **BF16-preserving short expert/
finalization alternative or attribution**, coordinated with its native owner.
It is not a ready restoration. Tiny/Tiny2 share helper0 geometry but bundle
required gate/up/down/gamma rounds with reader publication, cooperative launch
and PDL policy changes. The native owner confirms WGS-only MFIN/SC6/YBF
cannot explain these short cells. No minimal safe large restoration is yet
proved. Its proposed immutable effective-BF16-gamma precomputation preserves
the ABI, but lacks measured headroom and cannot honestly be nominated as a
>=1% candidate.

Within dense/router/attention the largest already-supported safe alternative
is the prior owned128 QKV+O group, only+0.37966% full-score cross-run sizing,
with no individual QKV attribution. It does not cover the short or5.77% gap.
The whole-four-cell hybrid is a separate official measurement, not evidence
that a precision rollback helps; its result must be interpreted independently.

Do not spend a job restoring pr8 identity, GLE's moved rounds, rs128 unchanged,
or removing an Lt guard that currently rejects no plan. For a new short expert
alternative first identify the actual selected source/flags and same-input
component/cold-path cost, preserve every required BF16 edge and progress proof,
then use independent controls/nulls and both short contexts plus all six loads.
Require actual routing/fresh/empty replay coverage, math/cache/guard checks,
resource admission and a material whole-score budget beyond noise. Stop if
the claimed gain depends on deleting a required round, unproved ordering or
only unmatched profiler subtraction. This report authorizes no new code/job.
