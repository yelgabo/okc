# Standalone MoE weight-reuse audit and diagnostic registration

October31 PDT, after the user deferred prefill. Evidence pin `031b94f` plus this
report's commit. Scope is standalone MoE only; no new GPU job or production change
is authorized by this document. Root owns shared index/publication and dispatch.

The current owned skinny path at1/8/16/32 and persistent47/51 at128/256 remain the
submission recommendation. The latest alternative failed: standalone masked
`ae8993f4e3e1435c` passed official tests and all40 eager/32 graph mathematical checks,
but lost145–184us at128 and203–235us at256, versus duplicate nulls within2.69us.
Its G1+G2 alone took380.511us in the256/std292 profile, exceeding the owned217.152us
whole-call median. This closes the tested default grouped schedule and glue-only
repair; it does not close all TMA/dequant architectures. Exact runtime configs,
counts and padding were outside the retained16KB tail and are not invented here.

## Measured opportunity and score coverage

`production51.7631974774944a2a` passed2.193073x,95%CI2.160177–2.225969.
The latest owned repeat `ae899...` judged the unchanged control at2.200397x,
95%CI2.167391–2.233403; neither establishes a new kernel gain.

| tokens | representative official median us | stated floor us | prepare loads-only us | prepare computing us |
|---|---:|---:|---:|---:|
|1|28.0|11.4|39.9|41.6|
|8|72.4|37.3|69.7|70.9|
|16|96.2|52.5|105.4|106.3|
|32|133.3|70.8|146.7|149.2|
|128|192.8|101.3|not in this skinny comparison|persistent51|
|256|241.8|108.9|not in this skinny comparison|persistent51|

Sources are the owning763197 receipt/log and `muse_moe_w4a16/TASK.md`. Prepare uses
different synthetic routing than the official medians. Loads-only still includes
the original schedule and synchronization and is not a pure HBM benchmark or a
rigorous arithmetic upper bound. It shows little demonstrated arithmetic headroom
on those inputs; it does not separate HBM, L2 delivery, ring pacing and helper waits.
Baseline hotspot source is `baseline_profile/status.eae9a6b7d5dd4809.json`.

The task scores six equally weighted logarithmic load ratios. A16/32-only change
covers2/6, as does128/256-only work. Small1/8/16/32 covers4/6. At root's23:26PDT
snapshot, rank10/best was2.595612/2.215712=1.171457. Uniform improvement confined to
two loads would require1.607605x at each (37.8% less latency); confined to four loads,
1.267914x at each (21.1% less latency). These are algebraic requirements at that
snapshot, not projections of any implementation. Floors are not attainable forecasts.

## Exact existing map

Source is the owned `submission.py`, `_SKINNY_BLOCK`, instantiated with SPLIT=1
by the host switch. `kUA=512`, `kUB=256`, `kUnitBytes=36864`.

1. Local counts are histogrammed independently in every CTA. Jobs are ordered by
   expert then eight-route group. Expert e contributes ceil(count[e]/8) jobs.
2. Phase A logical unit v has job=v//512, rowblock=v%512; its source is
   W13[(expert(job)*512+rowblock)*36864]. Phase B similarly uses256 and W2.
3. With G CTAs, CTA c owns [floor(total_units*c/G),floor(total_units*(c+1)/G))
   independently in each phase. Local q walks all of its A units then B units,
   with ring slot q%3 in the selected sk8w3s.
4. Repeated jobs of one expert repeat the same512+256 source payloads. Duplicates
   are separated by512 or256 global units, so neighboring CTAs at the same local
   step generally do not request the same payload. A multicast instruction toggle
   on the present mapping is therefore insufficient.
5. Each job must receive exactly512 phase-A completion credits. The helper fences
   its inter stores and batches credit per job before waiting for phase-B partials;
   compute acquires done_a[j]>=512 before reading inter. The final grid barrier and
   alternating generations also remain required.

`tools/moe_weight_reuse_model.py` reproduces these logical mappings on CPU and
models a candidate consecutive-job pair mapping. Each cluster owns a contiguous
range of pair-units; ranks0/1 process the two jobs at the same rowblock. Equal-expert
pairs may multicast one source; unequal pairs each unicast; only a final missing job
is dummy. Tests conserve every(phase,job,rowblock) exactly once and512 A credits/job.
The model is not a synchronization/progress or performance proof.

Synthetic bounds, not observed routing: eight local experts each with16 routes yield
50% logical duplicate requests; each with32 yields75%. Two-CTA pairing saves at most
50% of requests in either example; additional duplicates remain. AtM<=8 no expert
has more than one eight-route job, so this mechanism has no reuse opportunity there.
No fresh GPU count histogram has been collected for this hypothesis.

## Primary instruction support and open ordering/resource proof

[PTX8.7 section9.7.9.25.4.1](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
supports ordinary linear global-to-shared bulk multicast, including a CTA mask.
The original36KB packed unit is a multiple of16, with aligned source/destination;
no tensor descriptor or numerical repack is needed. The source and completion
signals target matching CTA-relative shared offsets. The instruction is advised
for sm90a, the target used here. This is source support, not measured benefit.

The current mode0 slot retirement uses a local CTA barrier and thread0 proxy fence
before its own refill. That alone cannot authorize an issuer overwriting a partner
CTA's slot. A new multicast writer must know BOTH recipients have retired that
slot and armed the next full-barrier transaction. Initialization must be cluster
visible before any remote completion. Keep all original fences; do not infer an
ordering proof from tests. A new release/acquire/async-proxy chain needs independent
review, including the last missing-job dummy and asymmetric consumer pacing.

The manual grid barrier requires all participating CTAs resident. A two-CTA cluster
can lose usable capacity at GPC boundaries even when nominal SM count is132. Before
any full fused prototype, query actual function attributes and
cudaOccupancyMaxActiveClusters with the proposed cluster geometry. Never assume66
resident clusters. Reduce G to2*actual_active_clusters if needed and account for the
lost capacity in the comparison. The copy-only diagnostic can avoid a grid barrier.

## Bounded diagnostic proposal, not yet dispatched

First collect held-out layer/std local counts at16/32 (two fresh seeds each), using
the fixed previously calibrated layer router8. Print all32 counts, jobs per expert,
logical requests/unique payloads, adjacent-CTA match count, proposed paired match
fraction and per-CTA work imbalance. Use the CPU map to check GPU-derived counts;
do not cache input-derived metadata in any timed challenge kernel.

Logical duplicate request bytes are NOT saved HBM bytes. Repeated requests may
already hit L2; multicast may reduce issue or interconnect pressure instead. Only
measurements can distinguish those. Without hardware counters, label physical HBM
and L2 traffic unmeasured.

If logical pair savings are material, a separate copy-only probe may compare:

- the current contiguous unit map with unicast;
- pair-aware map with unicast and an identical duplicate;
- the same pair-aware map with multicast and an identical duplicate.

All variants use the same36KB payloads, three-slot capacity, original packed
weights and consumer reads. Pair unicast/multicast share the same new retirement
handshake; otherwise the comparison conflates synchronization and copy mechanism.
Validate copied data/checksums for every recipient, not just time. Include cold-L2
randomized paired rounds, symmetric pacing and a bounded asymmetric-consumer case.
Print actual register/shared-memory/cluster residency and reject unsupported
capacity. No lowered precision, skipped fences, or production path in these probes.

Prospective early gates for root review: pairable request fraction at least10% on
both16/32 layer and std cases, then copy-only benefit at least10% and greater than
2x duplicate-null magnitude consistently. These are screening criteria, not a
promise of aggregate gain. Even a copy win requires full fused-kernel correctness,
progress and whole-score checks before promotion. Stop if the new mapping/handshake
cost erases benefit, capacity forces material underutilization, or fresh routing
lacks repeated payloads. Do not spend remaining checks on a full rewrite first.

No new GPU dispatch, kernel change, promotion or scored submission accompanies this
audit. Two CPU model tests pass. Root must reconcile this report with newer board
and job receipts, update the shared index, and publish through the normal workflow.

## Dated addendum: October31 PDT, fresh count screen fails

Root subsequently authorized ONLY a fresh-count check, not a copy/multicast kernel.
Check `53c07507b9fd4f8c` completed5/5 official tests; unchanged control2.201850463x,
95%CI2.168822706–2.234878220. All98 original top-level AST nodes were identical to
production51; CRLF normalized toLF in generation. No numerical/kernel/policy change.
Fixed router8 and held-out311/312 were used. All requested counts/map rows survived
the receipt tail, with actual deviceSM count132. Sources are committed
`profiles/weight_count.53c07507b9fd4f8c.{json,log}` under the MoE directory.

| tokens | routing | seed | jobs / active experts | logical duplicate requests | pairable request savings |
|---|---|---:|---:|---:|---:|
|16|layer|311|6 /6|0%|0%|
|16|layer|312|9 /8|11.1111%|0%|
|16|std|311|10 /10|0%|0%|
|16|std|312|12 /12|0%|0%|
|32|layer|311|11 /8|27.2727%|18.1818%|
|32|layer|312|11 /7|36.3636%|18.1818%|
|32|std|311|20 /20|0%|0%|
|32|std|312|20 /20|0%|0%|

The preregistered>=10% pairable savings criterion fails6/8 cases. Consecutive pairing
misses the sole16/layer312 duplicate because of pair alignment; an improved pairing
would still find no duplicates at all in the other16/layer case and all four std
cases. Original neighboring-CTA same-step matches are0 in all eight cases. Every
32-element count vector is in the owning log, not inferred from a maximum count.

Disposition: park the registered standalone16/32 multicast/weight-reuse implementation
before any full rewrite or copy-only GPU probe. Do not narrow to32-layer after this
failure and call the original gate passed. This closes the tested cross-routing,
two-size opportunity screen, not every conceivable cache/layout optimization.
Physical HBM/L2 traffic, multicast gain and cluster residency remain unmeasured.
No new copy kernel, fences, barriers, memory-ordering change or public promotion.
Reopen only with materially different measured routing/traffic evidence or a new
score-covered mechanism, not a repeat of these count seeds.
