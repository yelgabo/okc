# Attention ROI: stop funding boundary tweaks; pursue a step change

November2,2026,11:22AM America/Los_Angeles. Current human direction: quantify attention's full-layer contribution, reprioritize the captain toward top100 if it is small, update steering so workers can pick up work, and **"think about 10x improvements and not tiny improvements like this."** Evidence pin **8a05389d1052ac5a01090965e1e5e17b03775299**, source handoffs **19ab26cd /6e181164**, unchanged fresh layer **cc474883117947c0**, public incumbent **607eb52bdaf24160**. This supersedes the earlier attention-next priority and the prior1% admission floor as sufficient justification for campaign priority. It does not change any executed experiment's frozen statistic.

## Decision now

**Deprioritize attention continuity as a top100 driver. HOLD target checks, harness construction, further readiness work and tuning for this lane.** Preserve the delivered source checkpoint and its findings. This is an ROI decision, not a measured attention loss or a proof that every continuous pipeline is small. There is no demonstrated exposed-time budget supporting a material layer gain from this implementation.

The user-authorized reprioritization was sent directly to the existing **OKC Developer** chat, `01a19976-9e99-79c0-bab1-644bd8c0293a`, using the app tool. The worker explicitly acknowledged: attention stopped at published6e181164, no more readiness/harness/tuning/target work, and ready for the architecture-first source task below. This is actual direct coordination, separate from the repository handoff. The separate captain's roadmap is changed through this shared steering entry; no direct acknowledgment from that captain is claimed.

No target attention job exists. The worker's final candidate is **e912a5f4378301d62cd978b5012bc5d334dd0894f135784cf0c6cb50c87d6631**,872908 LF bytes, native4efb6714. Its report records7 CPU/facade tests and fixed secondary compiler comparisons, with preparation repaired to build both modules before capture. Those establish source progress, not target correctness or savings. [Worker report](2026-11-02-worker-attention-continuity.md) and archive remain intact; no native or production file was edited by this audit.

## What attention continuity could contribute

The treatment affects **full-context D16,D32,V16,V32 only**, with **16/62** of each workload's sliding/full weight. It preserves bulk attention math and KV traffic; the named mechanism carries a producer ring across request boundaries instead of draining/restarting per segment. Selected plans remain264 flat CTAs. Fresh complete-layer endpoints and profiled activities are:

|Load|Weighted complete-layer endpoint us|Full attention activity us|
|---|---:|---:|
|D16|152.016|88.6|
|D32|191.898|171.4|
|V16|261.713|104.9|
|V32|351.080|185.2|

For an equal **net full-cell endpoint** saving `s` in all four cells, the six-load score factor is `product(C/(C -16/62*s))^(1/6)`, with other loads and baselines fixed.

|Net saving per affected full cell|Whole-layer score gain|
|---:|---:|
|3us|0.239%|
|5us|0.399%|
|7us|0.560%|
|10us|0.802%|
|12.431us|1.000%|
|36.341us|3.000%|
|76.154us|6.576%, today's sampled layer100 gap|

The source comment at incumbent3295-3297 describes approximately7us of boundary cost at q_len8. It is historical commentary, not a fresh measurement, forecast or ceiling. The7us row generously illustrates that saving7us on the **entire call in all four cells** still yields only0.56% of the layer score. The treatment has not demonstrated that saving.

Conditional reductions of **all** the profiled full-attention activity, with every microsecond exposed and replacement costs already paid, yield10% activity cut ->1.048% layer,30% ->3.235%,50% ->5.553%. Covering the current gap would need about58.47% of those activities converted to net savings. These are sizing scenarios, not available slack. A boundary scheduling change has no evidence for removing most of the whole attention call.

### Why the mechanism deserves less priority

For contiguous partitions, each request boundary can cross at most one partition. At batch16, at most15/264 CTAs can span a boundary; at batch32, at most31/264. Existing SEGK400/600 virtual padding already gives crossing CTAs fewer real keys to compensate boundary cost. Proposed decode merge storage adds4608 shared bytes. Relevant incumbent anchors:172-173 SEGK;1389 segment drain;2639 merge scratch;2926 reuse join;3294-3299 partition padding;3401-3411 mapping;3444-3453 segment construction;3473-3479 final drain;3773 `_plan7`;11235 `_launch7`.

**CTA fractions are not timing bounds.** A boundary CTA may lie on the critical path, and init/tail/resource changes can affect single-segment CTAs too. Actual scored CUDA-generated sequence lengths and segment counts were not archived; CPU RNG replay cannot reproduce them. The sound conclusion is insufficient demonstrated ROI, not a strict mathematical less-than1% ceiling. Completing the source screen does not turn it into the best experiment.

## Revised work queue: dominant computation and complete alternatives

Top100 is the release objective. At11:22AM, layer100 is2.2659x versus public6072.126082676x: **+6.5763%** score required, before further board movement. Overall/layer are both366 in the independently queried boards; overall supplies ranks only. A plan should leave margin above the moving cutoff. No current candidate has an evidence-backed top100 payoff.

The user's10x direction changes **how we generate designs**: ask how to eliminate or amortize an expensive pass, transform immutable operands once when allowed, change computation/ownership across several workloads, or replace a complete engine. Do not spend another round improving an already small boundary merely because the source is ready. A10x claim about a tiny subroutine is insufficient; map its scope to the complete layer. Preserve the problem's numerical, routing, precision, memory-ordering and lifetime contracts.

The broad long-token expert engine covers D16/D32/V16/V32 in **both** contexts. If net endpoint reductions equaled10% of its current expert activities, the conditional layer gain would be **3.428%**;20% would be **7.170%**. Reaching the current cutoff needs approximately18.47% under that model. A hypothetical10x faster engine in that scope maps to **48.45% layer gain**, illustrating why a dominant-engine question is a better place for ambitious design work. None of those percentages proves such a design exists or that the activity is fully exposed.

### P0 — captain: preserve the pivot and own the next decision

Existing captain/coordinator owns the roadmap and sole judge-dispatch ownership remains unchanged. Index the delivered attention artifact as **SOURCE SURVIVOR, TARGET HOLD, LOW-ROI PRIORITY CLOSED**. Do not turn it into another source-validation or four-arm-harness project. Do not reopen exact pre-gamma,T1three-launch,G224,SiLU,serial,RMW or the closed outgoing seam. Existing PR setup failure remains a host-standard failure, not a compression result; unused quota grants no retry or public substitute.

The next target allocation must test a mechanism in a quantified path beyond today's approximately6.6% gap. Prefer a design with roughly10% layer opportunity/headroom; a first distinct piece should have credible several-percent contribution, or provide decisive information about such a larger plan. Compare against the strongest alternative and include replacement/integration costs. The previous1% component gate remains an experiment statistic, not a roadmap selector. No test is newly authorized by this report.

### P1 — existing OKC Developer: one45-active-minute architecture decision, no quota

Pick up this task from the stopped attention checkpoint. Own a **new report/evidence only** for a dominant expert-engine alternative; preserve current kernels and archives. No new target harness, kernel implementation, compiler sweep, installs, GPU/Driver/CLI POST, public/default action or child workers. This task ends at a concrete design decision or scoped STOP, not a helper optimization or another readiness framework.

1. Pin the complete legal control `ba9e327c`, actual MID(T16),B4(T32),MAIN(T128/T256) dispatch, input/weight packing, preparation and per-expert normalization/finalization. Treat each route separately; do not inherit a WGS result into MID. Start from the complete implementations rather than deciding to graft a loop into ours.
2. Inspect the **complete Humming W4A16/grouped-expert execution path**, from exact immutable source, as an alternative engine/dataflow. Establish whether Hopper NVFP4/BF16, scale semantics, zero cases, required per-expert normalization, expert sharding and actual tile sizes are supported before crediting a speed path. Humming's support of many formats is not proof of this combination. Reuse prior whole-G0 and MonoMoE scope audits as controls/negative evidence; do not repeat an unchanged donor census or call a known ring/deferred-bookkeeping feature new. No foreign code is executed, copied into a release or adopted by this task.
3. Produce **two materially different main-computation designs**, then select one by complete cost/coverage. At least one must challenge the full weight-processing/GEMM dataflow or engine base, rather than only epilogue transport, stage count, helper flags or finalizer ownership. Explicitly answer the10x question for the expensive subpath: what necessary work disappears or is amortized, what remains, and what that means for the whole call. Operation/byte/instruction counts establish work, not exposed time; distinguish those from a supported latency estimate and uncertainty.
4. Deliver a concrete selected-route before/after dataflow, full replacement costs (extra launches, traffic, preparation, scratch, register/shared resources, normalization, conversion and integration), and a six-load score budget. To cover the sampled gap with the long-expert scope under the conditional activity model requires net weighted load savings around D16=12.85us,D32=13.93us,V16=27.00us,V32=36.45us; recompute exact values in evidence. A3% first piece corresponds to about6.12/6.63/12.86/17.36us. These are required savings, not assigned performance.
5. Name the smallest later measurement that could distinguish the proposed main-computation win from unchanged work, replacement cost or hidden overlap, with controls, source pins, expected effect and fixed stop rule. Do not build or run it now. If neither alternative has a credible substantial layer path, return **NO ADEQUATE PATH FOUND** with the exact semantic/work/resource obstruction and the strongest next alternative. Do not quietly reduce the task into another ready1%-component tweak.

The45minutes bound this architecture decision, not the entire campaign. Captain must select the next broad alternative on the result; STOP closes a proposed bet, not the user's top100 objective.

### P2 — existing researcher: independent feasibility/ROI review, distinct ownership

Review P1's whole-operator equivalence, selected paths and budget rather than duplicating its implementation or a broad repo inventory. Challenge unsupported10x claims, count required expert-wide norm and BF16 roundings, and verify alternative preparation/packing and complete callable costs. Return ACCEPT FOR ONE SPECIFIED NEXT GATE or REJECT WITH COUNTEREXAMPLE. No new worker/chat or judge job is created. If this existing owner is unavailable, captain performs the review; do not invent an acknowledgment or restart stopped chats.

Broader short-chain or immutable-operand transformations remain reserves only if their **complete** scope can carry material layer benefit. Historical G0's5.77% and T1/T8 substitution's4.32% are confounded/counterfactual, not recoverable budgets. PR-only compression has only two-load scope and cannot be declared the whole top100 plan. Known repeated-weight-request census(~0.63%), fixed-owner finalizer(~0.34-0.43%) and stock dense substitutions(~0.42%) do not earn replacement priority without a genuinely different complete mechanism and new evidence.

## Evidence, provenance and current constraints

[Reproduction script](evidence/2026-11-02-attention-roi-pivot/analyze.py) reads immutable own receipts/source at8a05389d and validates the delivered candidate identity. [Analysis](evidence/2026-11-02-attention-roi-pivot/analysis.json) contains exact scenarios, thresholds, source pins and warnings; [fresh snapshot](evidence/2026-11-02-attention-roi-pivot/snapshot.json) retains boards/quota/jobs. Two read-only audit subagents independently checked planner scope, score arithmetic and alternative closures; neither implemented or dispatched work.

Primary sources re-read for the alternative-engine scope: [Humming's first-party architecture description](https://vllm-project.github.io/2026/10/19/humming.html) describes configurable data movement, dequantization and execution pipelines; it supplies no Muse speed result. [Pinned complete MonoMoE design](https://github.com/flashinfer-ai/flashinfer/blob/ce9f587b03f111b7a3aca8c9247bc4893653af15/docs/design_docs/monomoe_kernel.md) is a different FP8/fixed-shape operator; its headline and co-residency statements cannot replace a Muse eligibility/progress proof. These are known primary leads, not newly discovered qualified candidates. Prior own research already explains their transfer constraints.

Live state11:22AM: **layer0 private/5 scored, MoE8 private/5 scored, attention10 private/5 scored**, no live jobs. No check or submit was issued by this audit. Do not spend available attention capacity solely because layer capacity is exhausted. Current incumbent, numerical/order requirements and external-code policy remain. Subsequent result receipts or human instructions supersede this snapshot.
