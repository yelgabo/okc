# Exact-paste reconciliation and current fusion opportunity

November 1, 2026. Evidence pin `74f3091`, production snapshot `6da000f`, layer
candidate `muse_decode_layer/cand/layer_attn31s.py` raw SHA256
`585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b`.
Prefill remains deferred by the user. The active top-10 objective remains unmet.

The preceding goal turn confirmed the source identity but did not advance a kernel or
measurement. This continuation reconciles completed receipts, closes the losing wait-policy
hypothesis and implements a new measurement collector before choosing another layer change.

## What the pasted report establishes

This concerns [the exact preserved paste](evidence/2026-11-01-pasted-top10/reaching-top10.txt),
not the earlier community audit. Its central low-token whole-layer recommendation is a
reasonable hypothesis. Its 8–18% layer opportunity is an engineering estimate without our
latest code or measurements; it is not a measured portfolio, statistical interval or forecast.

[Primary-source provenance](2026-11-01-pasted-source-provenance.md) resolves its important
external examples: the 139 MB/58% profile is an author's H100 result; 1.5186x projection
belongs to Nathen Fernandes, and scored 1.331x belongs to team zzj. None is our current H200
implementation. The H100 table cannot independently reconstruct the claimed whole-graph
utilization without its graph duration and raw artifacts. Low utilization alone does not
prove removable activation traffic or exposed launch costs in our implementation.

## Current interfaces and already implemented work

Line anchors below refer to the pinned layer candidate, rather than a different standalone
implementation or a future generated diagnostic.

| Boundary | Existing path | Remaining hypothesis and required proof |
|---|---|---|
| QKV to attention | `kernel_fn:20997` materializes BF16 QKV; `_attn_path:21004` consumes it. Attention already performs QK norm, RoPE, direct FP8 cache writes, attention, output RMS and gate internally. | Projection-epilogue fusion is unmeasured. Preserve BF16 rounding before head RMS, cross-CTA head completeness, cache publication and existing barriers. M-C was an O/boundary experiment, not this fusion. |
| O to post-attention boundary | Separate O projection at21005 followed by `_abnd`/SGLang at21008–21014. M1 is selected conditionally. | Prior M-C `5ae9ce318e014edf` lost1.16us in9/9 before illegal access; not evidence of a safe winning implementation. No blanket claim that every fusion is closed. |
| Pooler/router/top-k | `_moe_front:19976` supports one-launch skinny pooler/router, concatenated GEMM, and separate router overlap. `_moe_front_route:20036` calls fused routing/pre-expert normalization. | Fusion can remove overlap or add a join. T1 whole-front variant already lost2.08us across9rounds; fresh T1 route-fusion gate failed. Do not repeat generic T1 fusion absent a new measured mechanism. |
| Routing to expert metadata | `_route_meta:17139` returns IDs/weights/local_start directly for fused MoE; no separate Marlin alignment there. Persistent CTAs build their own routing work. | Producer-emitted queues could reduce duplicated work, but require measured prologue cost plus publication/init/PDL proof. Standalone MoE inputs contain IDs/weights, not upstream logits/weights. |
| Experts to aggregate | `_experts_run:17193`, `_moe1_run:17390`, fused final pass near8799. Existing persistent pipeline fuses dequant, G1, SwiGLU, G2, expert norm and routed sum. | Whole-expert persistence is already implemented. A literal CTA per expert can leave most of132SMs unused at T1. Preserve per-expert RMS before weighting and the aggregate BF16 rounding at8831–8835. |
| Expert aggregate to unpooler | Unpooler is a separate projection. T<=8 already computes only1024 live output columns at21026–21032. | Intermediates are small; mandatory weights and full-row reduction dominate feasibility. Distributing unpool across experts changes rounding/nonlinear order unless equivalence is proved. |
| Unpooler to final boundary | `_bnd:19930` consumes live BF16[T,1024] and FP32 residual directly; it fuses final norm/residual/next pre-norm and avoids explicit zero padding. Above8 tokens the full8192 output remains. | Further fusion needs global row-reduction/ownership analysis. The norm still uses8192-channel semantics including zero columns. A larger CTA tile can sacrifice occupancy and bandwidth. |

At T1 the QKV tensor is4608B, expert aggregate8192B and short-unpool output2048B.
These are shape-derived byte counts, not measured HBM traffic. They do not justify an
8–18% forecast by themselves; weight traffic, launch latency and exposed dependencies matter.

Attention already has distinct decode/verify and sliding/full plans, grouped-query reuse,
eight-position verify reuse, direct cache conversion, producer/consumer specialization and
fused output normalization/gating. The layer chooses owned or fallback paths through gates;
do not claim every shape necessarily uses the owned plan after every preparation.

## Measured evidence that changes the next action

The latest layer public best is `8f2780d7382047e6`,13tests passed,2.022870479x,
95%CI1.992527–2.053214. Re-running `tools/hotspots.py` against its receipt and SGLang
baseline-profile `7d42771c3384449d` confirms MoE as a large hotspot, including33–35us
at decode1 and73–81us at verify1. Decode1 pooler is14.2–14.5us, QKV about9us, and full
attention12.3us. These are profiled kernel durations, including any device waits, not
additive exposed costs. The second paired layer-port study `9414182118414889` remains
unresolved under a different prepared M1 policy. The earlier sparse0.14 relative event has
not been explained; the separate matched diagnostic did not reproduce it.

The final skinny full-slot wait-policy check `9be599cf333a47c4` has complete16case data,
64 eager and64 graph checks, zero spin failures and exact four-arm resources:130registers,
0local bytes,155520B dynamic shared memory,1CTA/SM. Its two candidate contribution ratios
are1.000956258 and0.998638858 against null1.000295776. Both the>=1% improvement gate and
regression gate fail. Official2.207723314x (CI2.174607465–2.240839164) is the unchanged
owned control, not a new candidate score. Close this exact replacement; production unchanged.

Attention attribution `8ac6563343f74bcd` shows removing consumer math alone does not improve
verify/full; the producer/cache-copy schedule deserves source investigation. Perturbation
times are nonadditive. Prior mode51 pair weighting already lost (`7631033a0c93485d`,
`4ff339332fef4aff`); do not repeat weighting as a new idea. The corrected VT alias bound
requires119392B per CTA for four stages, too much for2CTA/SM. No invalid occupancy estimate
was used in a GPU experiment.

## Ranked next work and gates

1. **Measure current layer dependencies across all12 shapes.** New collector
   `tools/dl_make_layer_timeline.py` preserves the candidate source as an exact prefix and
   embeds the repository SGLang baseline uncompressed, with independent weights/state/input
   clones. It uses published soft-cache/agentic inputs and the FP32-front busiest EP group;
   saves/restores TF32; records candidate policies and actual local routing. Each arm has
   five cold event timings and three isolated, cold graph traces. The cache flush and input
   reset occur outside the profiler. Retain all24rows, full names and stream IDs through
   bounded lossless log encoding. A missing case, deadline, invalid shape/dtype/nonfinite
   output or over-budget report fails closed. This is diagnostic, never a scored submission.
2. **Choose a single exposed boundary from that evidence.** Prioritize the six-load
   `46*sliding+16*full` objective. Activity union is not SM utilization; overlapping kernel
   durations cannot be summed into removable time. Stop a proposed fusion if its plausible
   exposed budget is below paired noise, its interface changes rounding or its resource
   cost removes useful concurrency. PDL wait durations require causal interpretation.
3. **Attention producer source audit.** Reopen only a specific mode51 copy/address-latency
   mechanism with evidence distinct from prior L1 tests and mode51 weighting losses. Retain
   every fence, acquire/release, stage participant count and actual2CTA/SM allocation.
4. **Projection and MoE architecture pivots.** Codec/dense/ring/default-masked/wait sweeps
   remain closed as built. Weight reuse fails its screen in6/8 cases; all four standalone
   cases have no duplicate jobs. Reopen only with new scored-scope traffic or scheduling
   evidence, not a generic TMA/multicast recommendation.

Any future kernel candidate needs one hypothesis, independent duplicate controls, exact
official correctness and replay checks, measured resource vectors and whole-score improvement
beyond registered null/regression gates. Revert losers; retain receipts and closure logs.
Scored submissions use pure owned files only. Available quota is not evidence of a benefit.

## Feasibility and best candidates

The latest archived official snapshot remains November1 00:05PDT: projection gap5.02%,
attention11.73%, MoE17.15%, layer17.82% to the observed rank10 values. These are dated
aggregate gaps, not per-shape cutoffs. No current measured portfolio establishes those gains;
top10 by November4 remains unproven. Competitor per-shape receipts and physical HBM/L2/SM
counters are still gaps. The new timing collector does not claim to collect those counters.

Keep current validated production paths: projection C3, attention late-full2, owned MoE,
and pinned layer candidate585d46e2. Public bests remain C2 `9eba96894ee1402e`, attention
`8914751e3b9f468b`, MoE `abe7651dbaa44458`, layer `8f2780d7382047e6`. A lower fresh score
does not replace a public best. There is no new pure fusion candidate to submit yet.

Collector review, dispatch and measurement results are to be appended below with exact
job/source identifiers. No new layer performance result is claimed by this initial audit.

## Dispatch addendum

Collector source commit `097c38f`; generated payload984090B, SHA256
`5affd5cc9dd480af2a49f89db678cc989dfcd593fdb011a459330e618151abe9`.
Independent review cleared the reference argmax tie correction, baseline/input isolation,
capture/reset path and profiler API. Four tests pass under the existing WSL CPU-Torch venv;
the independently varied24row fixture encodes7952B, the35events/row stress fixture9752B,
both losslessly below the11KB log guard. Encoding is binary delta/duration plus interned
full names and streams; per-token local counts are recorded as an exact nine-bin histogram.
The generated source retains candidate prefix and kernel AST unchanged and is below1MB.

`okc check` returned exact layer job `41788b39fb824179`; direct status confirms running.
This is a new dependency measurement, not another fusion implementation or a public submit.
The preparation deadline is280s and incomplete reports fail closed. Diagnostic shape/dtype/
finite checks are not new mathematical certification; original gates and subsequent official
correctness remain relevant. No physical HBM/SM counters are claimed.

Source-only attention pivot disposition: mode51 page IDs are already prefetched three tiles
ahead (`attn_one.cu:1011`,1423–1430,1484–1504), before the empty-slot wait at1441. Moving
current-tile page-ID loads before that wait is not a new mechanism. The source audit is
motivated by `8ac6563343f74bcd` but adds no GPU experiment. Actual compiler issue scheduling
would need SASS evidence; address-offset arithmetic is a distinct unmeasured hypothesis.

## November1 00:53PDT evidence refresh

Fresh official board responses are archived in `evidence/2026-11-01-fusion-audit/`, with
observation timestamp and exact owning jobs in `summary.json`. Ranks are now294 projection,
208 attention,372 MoE and379 layer. Our scores and the four rank10 values are unchanged
from00:05; score gaps remain5.02096%,11.73079%,17.14577%,17.82229%. No problem is top10.

Attribution correction: the M-C row's initial15character identifier was missing its leading
`5`. Correct full job is `5ae9ce318e014edf`, verified against its owning decode-layer run row.
The measurement and closure are unchanged. The independent attention audit had the full ID.

## Collector observation failure and bounded correction

`41788b39fb824179` is terminal: preparation reached all24rows but the final encoded result
was12192B, above the11000Bguard. No trace rows were printed, so no latency, overlap or
fusion-opportunity inference is retained. No official score or correctness pass is claimed.
Raw status/log are preserved in `muse_decode_layer/profiles/timeline.41788b39fb824179.*`.

The corrected transport explicitly records trace coordinates at0.1us, while preserving
raw-resolution aggregate spans/union,0.001us cold event measurements, all event names/order
and streams. Binary encoding remains lossless for those declared recorded coordinates.
This changes measurement formatting only; kernels, weights, workload and prepared policies
are unchanged. Four CPU tests pass. Independent larger-fixture review precedes one retry;
another observation failure requires reassessing collection scope before more quota.
