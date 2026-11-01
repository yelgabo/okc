# Exact new pasted report: MoE/interface reconciliation

November1 PDT. Read-only audit; no new optimization or GPU dispatch. Evidence is the
new report `evidence/2026-11-01-pasted-top10/reaching-top10.txt`, normalized SHA256
f47bc1242e1a18e58afe82dc8d72bde6247f269e8771fa9781aedab029fffff0. This is NOT the
earlier community audit. Root owns combined-report/index publication.

The new paste explicitly recommends routing-aware persistent MoE, emitting expert
metadata from top-k, and one CTA/warpgroup per active expert (lines84–94). It estimates
4–10% low-token MoE latency gain, with an engineering range1–14%; its backlog row114
also estimates2–5% layer gain. Those are this report's estimates, not measurements or
statistical confidence intervals for our implementation. Its42%/81% HBM argument is
from a public H100 profile, not our H200 counters. Citation markers in the archived
paste are unresolved research IDs; this reconciliation does not independently verify
their external numerical claims.

## Map to exact owned interfaces and selected code

The explicit layer candidate is `muse_decode_layer/cand/layer_attn31s.py`, verified
SHA256585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b. Main
submission.py is a different older control. The release manifest and owning scored
receipt8f2780d7382047e6 identify the candidate/policies.

| New paste recommendation | Existing implementation | Remaining question |
|---|---|---|
|Routing-aware persistent experts|Standalone skinny and47/51 already execute routing histogram/job packing, W13 dequant/GEMM1, SwiGLU, W2/GEMM2, post-expert norm and route-weighted sum in one persistent launch. Layer `_moe1_run`17390 calls the same owned pipeline.|It is not a missing persistent conversion. A new scheduling/traffic mechanism needs evidence beyond the existing fusion.|
|Compact local work queues|Skinny groups expert-local routes into jobs<=8; fat uses jobs<=64 with output/stage units distributed across the SM-count grid.|Queue layout/metadata construction may still be improved, but not credited as absent or automatically expensive.|
|Emit expert metadata during top-k|Standalone Inputs already provide topk_ids and topk_weights, with no router logits/weights. The timed task cannot move an unavailable upstream top-k into this interface.|Only the layer exposes the actual top-k producer. It needs a separate interface/publication proof and measured critical-path gain.|
|Fuse routing metadata into expert path|For the layer's selected persistent path, `_route_meta`17149 already skips separate block alignment and returns raw ids/weights/local_start. The persistent CTAs construct routing/job metadata themselves.|A top-k-produced histogram/job list could avoid redundant per-CTA setup; not an already-proven gain. Race-free initialization, global completion, capture reuse and PDL ordering must be designed.|
|Fuse low-token pooler/router/pre-expert work|Layer `_moe_front_route`20036 offers skf-r/concat-r: front GEMM then a route kernel combining top8, pre-expert norm and optional hi/lo logit addition. `_moe_front`19976 also overlaps router/topk/metadata with pooler on a side stream.|Some proposed launches are already removed or overlapped. T1 selects fp32, and skf-r@T1 failed a fresh-input gate in this receipt. Do not book a generic top-k launch saving twice.|
|One CTA/warpgroup per active local expert|This is not the owned map: output-row units are spread over132 CTAs to fill the GPU even with few active experts.|Literal one CTA/expert covers at most32/132 SMs, and atM1 at most8/132. Need output/K parallelism or another concrete mapping; active-expert count alone does not establish adequate utilization.|
|Expert postnorm→unpooler→final boundary|Post-expert norm and route sum are already inside persistent MoE. A BF16[T,4096] aggregate still crosses to unpooler. ForT<=8 only1024 live unpool channels are computed, then `_bnd` combines final norm/residual/next-input norm without materializing7168 padded zeros.|MoE→unpooler and unpooler→boundary joins remain possible fusion boundaries, but require exact rounding, global reduction and resource analysis. The8192-wide zero-padding write is already absent on the selected short path.|

## Current selected policies matter

Owning scored8f2780d7382047e6 SUMMARY selects skinny sk8w3s atT1/8/16 and main51
atT32/128/256. Standalone7631974774944a2a selects skinny through32. Consequently,
the pending standalone skinny wait-policy check does not automatically improve the
current layer's32-token path. Selected front policy is fp32 at1, skf-r at8/16/32,
concat-r at128/256. These are owning receipt observations, not guarantees that another
prepare of the same source picks identically.

The layer's fused route producer computes sigmoid/top8/weights and pre-expert norm;
it does not currently emit the complete persistent expert work queue. `_route_meta`
deliberately bypasses old Marlin/W4 alignment when a fused variant is selected, so a
future producer-metadata experiment must compare against this selected path, not
against the older several-kernel alignment chain.

## Precision and ordering constraints on further fusion

The existing expert pipeline rounds GEMM1 gate/up to BF16, computes SwiGLU and rounds
its output to BF16, rounds GEMM2 output to BF16, then performs per-expert gamma/RMS
normalization and route weighting before the final sum/output rounding. Moving routing
weights before per-expert RMS normalization changes the operation. Distributing the
unpool projection over individual experts bypasses the existing BF16 aggregate
rounding and must not be assumed equivalent.

Standalone gamma is FP32 `(1+post_norm_w.float())`. The pinned layer obtains its
effective post-expert weight as BF16 at20855 and then converts those values to FP32
for persistent state nw at17989. This interface difference already caused a required
adapter correction before standalone masked checkae899...; do not silently transplant
the prefill/layer gamma boundary into standalone or vice versa.

Router precision and residual-stream FP32 boundaries also constrain fusion. The layer
has a distinct FP32 router input at small tokens and validated hi/lo alternatives;
reusing a BF16 pooled/front tensor without checking the original boundary and routing
gate is not automatically legal or accurate. New global expert metadata must be
initialized and published race-free across route-producing CTAs, and obey capture
lifetimes plus the existing stream/PDL and proxy-fence chains. No fence can be removed
on timing or passing-test evidence alone.

AtT<=8 the remaining two BF16 activation round trips are16KiB/token for the4096-wide
expert aggregate and4KiB/token for the1024-wide unpool slice (write+read). Their
nominal4.8TB/s raw byte-transfer equivalent is about0.0043us/token; this algebra is
not a measured cost or an upper bound on launch/dependency savings. It shows why low
whole-graph bandwidth by itself cannot prove these activation bytes explain several
microseconds. Weight traffic, launch latency, overlap, normalization dependencies and
occupancy require actual critical-path measurements.

## Reconcile this paste with measured failures

- Standalone default Humming INDEXED8d64807ea7504495 lost, and had a BF16-gamma adapter
  difference. Standalone GROUPED_MASKEDae8993f4e3e1435c corrected FP32 gamma, passed
  all40 eager/32 graph math checks, but lost145–184us at128 and203–235us at256. Its
  G1/G2 alone exceeded the owned whole-call latency. This rejects those tested
  schedules, not all persistent/TMA architectures or every low-token design.
- Fresh count53c07507b9fd4f8c found no repeated eight-token weight job in all four
  standalone16/32 cases.32-layer cases had pairable logical savings18.18%, but the
  registered two-size/two-routing screen failed6/8. No multicast/HBM benefit was
  measured. This closes that registered reuse recipe, not producer-metadata fusion.
- Existing loads-only≈computing and corrected full-wait counters leave wake/poll
  pacing versus physical memory bandwidth unresolved. They do not validate the
  paste's proposed gain percentages. Current wait-policy check is a separate,
  source-proven same-acquire hypothesis, not implementation of the paste's CTA/expert
  redesign; two host-harness failures currently prevent any complete timing claim.

The actionable conclusion is to mark existing fusion as done, preserve the remaining
layer producer/consumer interfaces as hypotheses, and measure selected whole-layer
critical-path gain before promotion. No new optimization beyond the previously
authorized wait-harness repair is proposed or dispatched by this reconciliation.
