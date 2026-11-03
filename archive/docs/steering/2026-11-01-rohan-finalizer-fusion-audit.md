# Rohan published unpooler/finalizer: source audit and full-stage budget

November1. Root evidence `7fc1975b05af38493c25d9952168256a18dc0dc5`;
owned report parent `31e8155`. No source adoption, kernel edit or GPU check.

**Decision: the source contains a real producer-side fusion, but does not establish
a conforming high-value replacement for G1b3.** Park implementation/check dispatch
pending a precision-preserving reduction design and exposed timing. Do not close
all producer-side fusion from the existing boundary-only1.9us argument: counting
every affected unpooler and final boundary gives a larger, albeit unattainable,
zero-work ceiling of3.5298%. The external implementation omits required BF16 rounds.

## Exact source and external evidence

Primary source: rohanarun/Auto-research-agent commit
`018b7a102348ea3b73701e8be52d24f82d8c91ee`,
[candidate](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates/gemini-3.8-flash-opencode/kernel-muse_decode_layer/submission.py).
Local `.build/today/round2-published-rohan/muse_decode_layer.py` SHA256
`2d9aa818d43bc1fba7c67734d15ce992853cff6a6f814669ff999fd47ff27b12` exactly matches
the published metadata. Header credits rohanarun for UB2/SK2B and the UV restructure;
it also retains team vjeux component provenance and Apache2 notice. No ownership
inference beyond those source statements is needed.

The [published evaluation](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates/gemini-3.8-flash-opencode/kernel-muse_decode_layer/evaluation.json)
claims two H200 banks and correctness. Its24 cells all have exactly2.1724 ratio;
each cell's five paired timing records are identical. These may be normalized
summary values; they cannot supply per-shape gains, variability or fusion attribution.
The benchmark results metadata dates the trial October30 and gives a curator audit.
Neither artifact is an official OKC job receipt. No ranking or official speedup
is inferred from these values.

## Reachable fusion versus the header shorthand

- UB2: `_ub2_prepare`13237 packs only the first1024 unpool rows, accepts bounded
  escape tables and requires32 active clusters. `_ub2_run`13259 is reached at14730
  only with UB2 enabled, T<=8, gluev2 off and successful preparation. The earlier
  UV branch14727 also uses UB2 but rebuilds s1 from older stream/O/normalizer inputs;
  `_uv_on`14988 additionally requires the bfpf front and sk2 O selections.
- SK2B:12501–12504 enable only **bucket16**, not32; `_sk2b_prepare`12519 also
  requires a prepared sm90 sk2 configuration and capacity>=grid. Call14722 requires
  8<T<=32, no gluev2, mm_pick un=bfp and actual kind sk2; bucket membership narrows
  this to9..16. It handles full8192 output columns. O+boundary1 fusion is disabled.
- Both return final BF16 hidden output and FP32 residual stream directly; neither
  caches activation outputs. Persistent sums/bar counters are scratch state.

UB2 keeps the unpooler BF16 rounding at13034 before multiplying gamma. Its1024 live
columns and zero tail still use the **8192** norm denominator at13100; final residual
norm likewise uses8192/eps1e-5, with branch eps1e-8. SK2B10470–10590 follows the same
full-width denominator and writes all8192 columns. Sliced denominator preservation
is real; this is not a1024-denominator shortcut.

However, both compute `f=BF16(unpool)*gamma` in FP32, keep `f*r1` in FP32 and form
the residual from it. G1b3's repaired `_boundary2`10845 and `_sl_cluster_finish`15642
require **BF16 rounding after gamma and again after branch normalization**. The
fusion omits both. Its final norm uses four moments to evaluate
`c²+2*r1*c*e+r1²*e²`, with c=alpha*s1/e=beta*f. That expression does not implement
the required BF16-rounded normalized branch; even without those rounds it changes
finite-arithmetic summation/cancellation. Passing an external audit is not a proof
of the repaired ABI. Keep the rounds; no tolerance relaxation is proposed.

UB2 uses128 CTAs in32 clusters, local/shared exchange plus remote st.async/mbarrier,
then a software global generation counter and replicated atomic norm sums. It
queries active clusters but launches ordinary cluster+PDL attributes13118–13158,
without a cooperative all-grid launch contract. SK2B10725 queries resident CTA
capacity but similarly uses ordinary/PDL launch paths. Their releases/acquires,
initialization visibility, last remote-reader lifetime and parity reuse require a
complete proof in an owned design; capacity alone does not guarantee grid progress.
UB2's expert input uses coherent generic loads after its wait; this is distinct
from the SK2B TMA input path, whose generic-to-async crossing needs the repaired
reader proxy fence. No barriers may be removed based on a passing comparison.

## Current coverage and all affected score weight

Current G1b3 source SHA256
`82ddbc1d4d1941aac78c129830deeb364c6932bdfa4e1709e1985101487478cc`
has no UB2/SK2B. T1/8 use a separate BF16 sliced unpooler and `_sl_cluster_finish`.
The preceding boundary already precomputes the residual tail; the finalizer repeats
the1024 live-head reductions for output tiles. T16 uses full unpooler plus separate
post-FFN cluster boundary. This is not already producer-side fused. G1b3 has no
receipt yet: sizing uses actual G1b2 `736a5cbc977f4ad4` with the same projection/
boundary source, **not a measured G1b3 latency forecast**.

|Load|Context|Weight|Unpool us|Final boundary us|Whole adjusted us|
|---|---|---:|---:|---:|---:|
|decode1|sliding|46|4.5|1.9|76.923|
|decode1|full|16|4.4|1.7|83.615|
|decode16|sliding|46|5.9|3.4|132.798|
|decode16|full|16|6.1|3.7|212.540|
|decode32|sliding|46|unaffected|unaffected|159.381|
|decode32|full|16|unaffected|unaffected|310.008|
|verify1|sliding|46|5.0|1.7|110.857|
|verify1|full|16|4.5|1.8|114.062|
|verify16|sliding|46|unaffected|unaffected|266.664|
|verify16|full|16|unaffected|unaffected|350.368|
|verify32|sliding|46|unaffected|unaffected|336.668|
|verify32|full|16|unaffected|unaffected|499.824|

Official aggregation is six load times `(46*sliding+16*full)/62`, then geometric
mean. Setting all six affected producer+consumer sums to zero yields1.035297788
(+3.52978%). T1/8 only yields1.024407649 (+2.44076%). Setting just final consumers
to zero yields1.010435527 (+1.04355%) across T1/8/16, or1.006580770 (+.65808%) for
T1/8 alone. These are deliberately optimistic arithmetic envelopes using one
retained profile sample; overlap, launch gaps, fresh seeds and new reduction cost
can invalidate direct subtraction. They are not attainable projections or physical
bounds. Arithmetic uses `counterfactual` from `tools/dl_proj_budget_audit.py`.

## New owned mechanism and decision gate

A distinct mechanism remains plausible: form norm partials **after the producer's
final FP32 K reduction and BF16 unpool output**, retain the live values locally,
then execute the correctly rounded residual finalizer. This differs from simply
combining two existing launches or mode79 expert finalization. It may avoid
intermediate global materialization/repeated live-head reads. Its benefit is not
established by the external source because the omitted BF16 rounds avoid work and
change the synchronization problem. A correct design needs branch-norm reduction,
BF16 normalized values, residual construction, and a second residual-norm reduction
or an explicitly costed redundant live-head computation. Do not assume the four
moments eliminate the second reduction while preserving the required ABI.

Before implementation, attribute exact producer completion, available norm partials
and consumer demand on G1b3, including resource/residency. Require an isolated owned
producer+consumer positive control; the entire boundary-only envelope is just1.04%.
The old persistent264-CTA O fusion `5ae9ce318e014edf` lost1.16us and remains a scoped
warning about serialization/global barriers, not closure of this1024-slice mechanism.

If a changed design clears that model, the true pair must share projection geometry,
weights, all BF16 casts, gate/eps/denominator, expert/frontend policy and final ABI.
Control writes its BF16 unpool then runs the current finalizer; candidate retains
producer outputs and runs its new reduction/finalizer. Four independently compiled/
private-scratch arms, two fresh routed seeds, cold reversed/randomized order and
both contexts are needed. Compare complete official twelve-cell score, including
all affected T1/8/16 cells and unchanged-shape regressions; require >2x duplicate
log noise, official correctness, direct boundary oracle including cancellation/
rounding cases, FP64 projection checks and alternating fresh/empty graph replay.

**Stop now at the source/model gate:** no compliant speedup has been demonstrated,
so this source alone does not justify a GPU check. Reopen with the exact precision/
ordering design and exposed budget above. Do not adopt external code, repeat the
old persistent-O arrangement, or close all producer-side fusion as impossible.
