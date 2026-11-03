# Why G0 is faster than our fork: unresolved integration cost

November1,2026. Root evidence checkpoint `9714e8f`; native inventory
`e8c4a4b`/ownerd2bdeec and precision inventory `9714e8f`/owner76ebb0f.
This responds to the user's question about reaching keell0renz's level
with its public code. It does not renew closed locality or MFIN experiments.

**We do not yet have a causal explanation for the remaining gap.** Our G
fork bundles necessary precision/ordering repairs with conservative policy
choices, and its jobs use different inputs, routing and baseline timings.
Calling the whole difference an unavoidable cost of compliance would be
unsupported. Calling every original optimization safe because it passed13
tests would also be unsupported. The discrepancy stays explicitly open.

## Three different observations

| Observation | Owning identity | Result |
|---|---|---|
| Exact public G0 file measured by us |8e30e1be upstream; rawSHA1ec189dc,995999B; check3a5f879df0d44825|PASS13,2.248790532x,95%CI[2.215058674,2.282522390]|
| Our repaired G1b4 stage file |SHAba9e327c,869076B; public607eb52bdaf24160|PASS13,2.126082676x,95%CI[2.094191436,2.157973917]|
| keell0renz team board row,20:12UTC snapshot |public job232de027eda1481f|rank76,2.2659x; not our exact-file measurement|

G0/607e official score ratio is **1.057715468**, or about5.77% more score.
The team row has another job identity; do not substitute its score for the
file measured in our H200 check. G0 itself remains below the current2.2935x
layer rank50 cutoff. Reaching G0's observed point would be useful progress,
not proof of top50 or top10. Overall's API supplies ranks only.

Root independently reconstructs both six-load scores from all12 benchmark
records. Its receipt hashes, raw/adjusted cells, seeds, per-job baselines,
weighted loads and board-row identity are retained in
`evidence/2026-11-01-g-integration-gap/cross-job-gap.json`.

| Load, (46*sliding+16*full)/62 | G0 us |607e us | Latency difference |
|---|---:|---:|---:|
|decode1|68.203874|78.549045|+15.17%|
|decode16|149.700710|158.241010|+5.70%|
|decode32|190.719032|191.739736|+0.54%|
|verify1|103.618625|115.990823|+11.94%|
|verify16|258.670706|269.347029|+4.13%|
|verify32|350.796973|359.297216|+2.42%|

Inserting only G0's two short whole-load times into607e while holding its
other four loads and baseline times fixed gives conditional factor1.043244918.
Inserting all six gives1.065243721, **not** the official1.057715468 ratio,
because the original jobs measured different SGLang baselines. These are
sizing scenarios, not causal estimates, attainable savings or new hybrid
scores. The short paths are the largest credible location to investigate.

## Prioritized upstream-to607e inventory

Detailed actual-source lines, transforms, primary baseline/rule references
and exact hashes are in the [native inventory](../../../docs/steering/2026-11-01-g-native-repair-inventory.md)
and [precision inventory](2026-11-01-g-upstream-precision-inventory.md).
The taxonomy below distinguishes the required property from our chosen
implementation. None of these rows has an isolated matched H200 price.

| Priority / actual change | Requirement versus choice | What the existing measurements establish |
|---|---|---|
| **1: short Tiny/Tiny2 expert path** | Geometry/helper0 retained. BF16 gate/up before SiLU, SwiGLU, scaled down result before gamma, effective gamma and reader publication were repaired; cooperative progress retained and PDL disabled. Required operator/protocol properties must remain. | No per-repair ablation. Positive short expert profiler-row substitution gives conditional1.056961678, versus dense1.0, attention1.000693403 and glue1.006726342. Different routing and summed/overlapping profile rows prevent causal or exposed-time conclusions; do not add the factors. |
| **2: global PDL off** | A conservative whole-pipeline reference policy, not a blanket CUDA/challenge prohibition. A particular edge still needs producer publication, consumer waits before reads, descriptor lifetime and resident progress proof. | No isolated whole-layer PDL cost. Enabling a lone launch flag may not restore a coherent overlap chain. Combined cooperative+PDL support is not established merely by finding both enum names. |
| **3: final-grid atomic strengthening** | Retained writer fences/joins and GPU acquire are required. Whether the added acq_rel arrival RMW is itself necessary is a separate PTX release-pattern question. | A new narrowly scoped source/compile investigation is described below. No H200 latency or speedup exists. This is not permission to remove explicit writer/proxy fences. |
| **4: native attention pr8 BF16 round** | Active later qualification change, not just the first attention patch. Baseline rounds attention output before RMS/gate; keep that boundary. | Full ATT7 delta is exactly this round plus PDL attribute0; CPP unchanged. No isolated cost. Positive all-cell attention profile sizing only1.0010015; it does not explain the short expert gap. |
| **5: ordinary SK2 proxy repair** | Two new fences execute only under if(pdl);607e sets PDL0. Dense arithmetic/decoding otherwise identical. | Those added fences do not dynamically execute in the selected policy. Do not charge their direct instruction cost to607e. Lost overlap is a different hypothesis. Short total dense is already faster in these profiles. |
| **6: MFIN0 / full join retained** | Original4M-iteration readiness fallthrough is unsafe. Distributed finalization is not categorically forbidden, but toggling the original flag is not a fix. | Main/B4 only; **not active for short helper0**. The fixed-owner128/256 frontier1c6ebc76 has optimistic whole-score maxima1.003445–1.004271 in its scope. It neither explains D1/V1 nor proves a universal MFIN ceiling. |
| **7: DF alias replacement atM16** | Original sentinel path has separate publication/readiness defects. Preserve the repaired helper protocol until a replacement is proved. | M16 cannot directly explain the T1/T8 discrepancy. No isolated DF-only H200 cost. |
| **8: YBF0 in WGS main** | Conservative removal of extra post-gamma quantization, distinct from required BF16 pre-gamma boundary. Internal-reduction rule scope needs precision-specific reasoning. | Inactive for short WS helpers. No isolated YBF price or certified rollback. |
| **9: rs128 off / nonsplit Lt guard** | Conservative internal-reduction policy, not an explicit universal all-partials-FP32 rule. A verified compliant reduction could be considered separately. | All eight607e Lt plans are installed; no rejected-plan fallback cost observed. Positive large-dense sizing1.0048455 is unmatched, not exposed saving or a ceiling. No short-gap explanation. |
| **10: deferred GLE_T / B1X off** | The old rewrites move BF16 rounds across nonlinear/residual/linear stages; carry reads also precede dependency waits. Fixing one pointer/wait is insufficient. Active FRT and FP32 short-router interface are retained separately. | No isolated cost. Re-enabling the old rewrite is not a minimal safe restoration. The ordinary serial FRT-versus-split question is already queued with full-graph gates. |
| **Retained: SC6, shape flags and stage retirement** | Lossless SC6 remains active for eligible WGS. Short helper geometry remains. Current-stage retirement preserves complete input read lifetime. | SC6 was not removed. Matched63f5a2a8 stage recovery1.018493–1.028699 is already in607e and cannot be counted as future gain. SC6 standalone05e51d failed repeatability/regression gates. |

The strongest required claims come from the pinned SGLang/operator interfaces
and specific memory-model obligations, not from observed numerical tolerance
passes. The [earlier precision-contract correction](2026-11-01-g-fork-precision-contract.md)
still applies: do not invent a universal ban on internal BF16 split partials.

## The locality hypothesis was tested, and lost at its tested scope

Direct same-expert fixed-band32 job**a5c7c7e3fe344919** passed5 official and
264 private checks, but all six targeted rows were slower; its four full-score
factors were0.942199–0.945964 versus control. Cached shared-band32
**903e5b03411c4e2d** passed396 private checks, but cached/control factors
0.994470–0.998904 failed the registered gates. Cached beat the failed direct
variant; that was recovery from a loser, not a baseline win.

These were real implementations of the locality/reuse hypothesis. Their
logical duplicate census did not establish exposed HBM savings. Close those
two exact representations and their knob sweeps, while retaining the G0
integration discrepancy as a **different unresolved question**. The invalid
f147 detailed observer also supplies no trustworthy phase/retirement cost.

## One minimal restoration under proof and compile gates

The specific next source target restores **only** the final-grid arrival's
strong GPU relaxed RMW in actual short Tiny/Tiny2, preserving the returned
old counter and compiler memory clobber. It retains all explicit writer
threadfences, proxy fences, named/CTA joins, GPU acquire polling, generation
cleanup, BF16 rounds, cooperative admission, PDLoff and MFIN/DFoff. MID,
main/B4 and all other cells stay unchanged. The default builder is OFF.

This is based on an explicit PTX release-pattern alternative, not passing
tests: writer-fence plus strong write can publish prior work, while observation
through the atomic chain and the final acquire must cover every writer.
Full CTA relays are important because compute-tail writers are not all the
helper lanes. Actual mutable finalizer loads must remain coherent generic
loads; the model does not cover ld.global.nc. [PTX ISA8.7 memory model](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model).
Independent proof and emitted-code review are required before qualification.

The emitted opt-in source is869551 B/d33915c7; local CUDA12.8 sm90a Tiny/Tiny2
retain147 registers,3 barriers and zero stack/spills. Normalized PTX differs
only at the arrival atomic. SASS removes the associated atomic-lowering
ordering/invalidation block; all explicit PTX fences, reader acquires and
math remain. This is a meaningful compiler delta, **not measured speed**.
It must not be extrapolated to the whole5.77% discrepancy.

A private four-owner comparison is being prepared with both arms based on
exact607e, including its existing stage-on WGS paths. It may become the next
direct G-gap experiment **after the existing first-two reset jobs**, not a
reason to reorder or duplicate them. The old owned four-cell comparison
remains held. Require exact compiled flag0/1 only at T1/T8, all-other-family
identity, natural routing equality, independent mutable scratch/Lt, fresh/
empty/capture/math/cache/guard/resource checks and complete six-load coverage.
Every held-out seed/four cross must yield>=1% whole-score gain, exceed2x the
duplicate log-null, and respect per-context<=max(0.5us,1%) regression. Stop
on proof, resource, capture, budget or coverage failure, or a valid performance
miss; no post-hoc winning-cell selection or subsequent atomic/fence sweep.

## Current operational decision

Keep607e as the layer recommendation. New four-cell public job**1fad70bbe3854cbc**
has unavailable outcome during API530 maintenance and leaves these two short
cells unchanged; it is not a demonstrated solution to the G0 gap. Reconcile
that exact ID before retrying anything, then preserve stage/short reset order.
The broader roadmap is now explicit: favor this proof-backed, minimally
changed short-expert question over an unrelated microtune once its gates clear.
No ready large precision rollback or promise of G0/top50/top10 performance
follows from this inventory.

## 21:12 UTC handoff: one concrete private check is frozen

Root integration checkpoint `6fe1134` now includes the
[scoped PTX/epoch proof](2026-11-01-g-ordering-restoration-proof.md),
[actual selected compiler/load census](2026-11-01-g-short-final-rmw-compile.md),
[private source review](2026-11-01-short-rmw-private-review.md),
[whole-host review](2026-11-01-g-short-rmw-host-review.md) and
[strict owning reader](2026-11-01-short-rmw-owning-reader.md).
The source change has an explicit memory-model argument and independently
checked actual lowering; no fence was removed on numerical tests alone.

Root double regeneration produces **906841 B**, SHA256
`f0426f7160768de01997b271744542ae3442d0851a1c27e586ced322a8fc6864`.
The private artifact is regenerated with
`python tools/g_short_rmw_check.py ROOT SCRATCH/check.py`; never submit it
publicly. Ready manifest and normalized dependency hashes:
`muse_decode_layer/profiles/short-rmw-ready.2026-11-01/manifest.json`.
Root reran11 actual source/score/host tests, PASS4.132s, plus the five owning
reader tests, PASS0.011s. GPU/native/reference arithmetic services remain
explicit CPU substitutes; these16 passes are not new H200 measurements.

There are **two compiled variants per short module and four private state
placements**, not four independent compilations. Both sides retain607e's
stage-on WGS/B4 policy and other native families. The pure opt-in follows
Tiny/Tiny2 families, including their valid smaller neighboring token counts;
the private scored banks cover T1/T8. Generic/public correctness remains a
separate clean qualification gate.

Coverage is36 median rows,144 normal checks,144 fresh-final checks,144 model
captures and32 empty checks. Empty precedes occupied work without manual
counter reset. Eight changed-input replays are checked at their **final**
output, not individually. The capsule retains medians, not24-sample vectors.
Final owner/guard checks occur after printing, so a capsule alone is never
success: the reader requires the own terminal private-check13tests/12bench/
12profiles/flags/errors/setup gates before interpreting it. Private ratios
are raw cold GPU-event whole-graph comparisons, not official adjusted scores.

The **same** existing16:00PST reset reminder was updated/read back, with no
new timer. It must first resolve accepted1fad70bbe3854cbc and fresh service/
live-ledger/quota; then preserve full-G stage first, serial front second,
and this private counter question next. Old owned-four-cell remains HOLD.
The last root probe still returned530 for status/jobs/quota. No additional
GPU job was dispatched. Code remains OFF and incumbent607e is retained.

On node-migration recovery, record reported GPU/node/driver/runtime,
evaluator image/code, baseline and compiler identity when available; mark
missing provenance explicitly. Same-runner paired controls are the causal
reference. Hardware/admission failures stay fatal. If1fad establishes a new
validated incumbent, reconcile it before promotion: f042 is a607-lineage
diagnostic and cannot by itself certify a combined hybrid improvement.

The separate standalone MoE progress prototype68bf4c1f stays OFF: root's five
CPU tests pass, but both native fatal representations fail main47's
no-new-spill gate. That is a recorded local static miss, not a GPU slowdown
or a reason to certify the old ordinary-launch source. It is unrelated to
the new G short-counter candidate's positive bounded compiler gate.

## 21:34 UTC correction and superseding user priority

The new [G integration-first plan](2026-11-01-g-minimal-delta-plan.md)
supersedes this report's old stage-first/serial-front-second reset ordering.
Unrelated tuning is paused. Original stage/short/f042 artifacts are HOLD
pending common inherited attention-progress and pre-expert coefficient
qualification; f042 remains the first relevant repair-representation question
after that common control. The app goal and its reset automation are paused.

Exact-source correction: G0 already rounds the SwiGLU output before GEMM2.
That boundary was preserved, not newly repaired. The table above bundled it
with newly added gate/up and pre-gamma rounds too broadly. Do not assign an
added SwiGLU-output-round cost to our fork. The full native ladder now
reconstructs both actual607 literals with no unclassified remainder; causal
H200 prices remain unmeasured. Shared inherited attention launch/pre-expert
coefficient issues are eligibility dependencies, not explanations for the
historical G0→607 difference.
