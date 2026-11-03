# T1: full-layer ROI and completed captain review

November2,2026,11:10AM America/Los_Angeles. User asks how much T1 can improve the full layer, whether another check would be a tiny optimization, and what blocks captain review. Evidence base **5f68780152ff6f33bc5e8c967e85d41b23e391e7**, own T1 target **28248eb5e5854041**, fresh full-layer control **cc474883117947c0**, public incumbent **607eb52bdaf24160**. This handoff records the current human ROI priority; it allocates no implementation or judge work.

## Answer and current outcome

**No captain review remains blocked. The review and one private T1 test have already completed, and this exact alternative lost decisively. No second check or promotion is allocated.** The earlier18:28UTC monitor snapshot correctly had a held proposal, but newer commits supersede that status.

The T1 mechanism is narrow, affecting decode1 only, one of six equally weighted workload groups. It can contribute meaningfully only if it makes a sufficiently large **net complete-layer** improvement on that load. Under the current fresh endpoint, **1% full-layer score needs4.565us net decode1 saving;2% needs8.824us;3% needs12.800us**. There was no measured exposed-time budget showing this particular three-launch replacement could save that much after its extra launches, intermediates and changed overlap.

One private MoE check was already spent by the sole worker under captain allocation `c66f5ce6`; it was not a layer or public submission. The allocation explicitly described exploration rather than a forecast of sufficient layer benefit. That bounds the cost but does not establish high ROI. The allocation's source/host readiness and its1% standalone component gate did not supply the missing full-layer benefit argument. Earlier [short-output-ownership research](2026-11-02-research-short-output-ownership.md) had already recorded the4.565us hurdle and missing exposed saving: this was an allocation/enforcement gap, not a missing written lesson. Do not append another general principle instead of applying the existing objective/ROI rules.

## Quantitative full-layer sizing

From own fresh unchanged full-layer receipt `docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json`, decode1 medians are77.036416661us sliding and83.732859275us full. Its46:16 weighted endpoint is:

`D = (46*77.036416661 +16*83.732859275)/62 =78.764530884us`.

Holding the baselines and other five loads fixed, a **net** weighted decode1 reduction `d` yields whole-layer score factor `(D/(D-d))^(1/6)`. The weighted saving permits different changes in the two contexts: `d=(46*d_sliding+16*d_full)/62`. One-cell savings must retain their46/62 or16/62 weight.

| Net weighted decode1 endpoint reduction | Whole-layer score gain |
|---:|---:|
|1us|0.213%|
|2us|0.430%|
|3us|0.649%|
|5us|1.099%|
|10us|2.289%|

The public607 receipt independently gives essentially the same1%/2%/3% thresholds:4.552/8.800/12.765us. This endpoint sensitivity is robust across those two receipts; neither receipt measures this treatment's exposed saving.

The fresh profiler reports T1 expert activity29.1us sliding and22.6us full, a weighted27.422580645us. **If** an expert-activity reduction translated completely into net endpoint saving after every replacement cost, the sizing would be:

| Hypothetical expert-activity reduction | Hypothetical whole-layer score gain |
|---:|---:|
|10%|0.592%|
|20%|1.210%|
|50%|3.239%|

These are conditional scenarios, **not predictions or recoverable ceilings**. Profiler activity and benchmark medians came from separate replay observations. The public607 full-cell expert activity is42.8us versus the fresh22.6us, illustrating why a profiler sum cannot be treated as fixed available latency. No exposed cooperative-wait, launch, scheduling or overlap cost was measured before the T1 allocation.

A1% standalone equal-six gain with only T1 changing corresponds to just5.795% T1 component latency reduction. Converting that reduction through the fresh expert-activity scenario gives1.589us net and only **0.340% layer score**. Therefore **the1% component gate was not a1% layer ROI gate**. Keep the executed frozen gate unchanged for attribution; add the missing full-layer screen before a future allocation rather than changing an experiment's threshold afterward.

For perspective, reaching the sampled layer-rank100 cutoff2.2657x from public6072.126082676x using T1 alone would require24.987us net on decode1,31.72% of its complete endpoint and about91.1% of the fresh profiled expert activity. This is a demanding conditional target with no demonstrated saving budget, not a rank forecast or a claim that every such microsecond is removable. Overall supplies ranks only. T1 was not an evidence-backed top100 solution by itself.

## Actual experiment and review timeline

All times below are November2 in America/Los_Angeles, UTC-08:00.

| Event | Time / evidence |
|---|---|
|Worker held source handoff published|10:23:22AM, `ce14d11f`|
|Captain independent review and one-check decision|10:31:43AM, `c66f5ce6`|
|Worker preregistration|10:36:06AM, `9d7937d2`|
|Own target result completed|10:43:21AM, `28248eb5e5854041`|
|Worker terminal archive|10:50:12AM, `1786dbf2`|
|Captain independent terminal verification and closure|11:04:06AM, `5f687801`|

The fresh service receipt exactly matches the worker's archived result. I independently reran its unchanged frozen reader against the fresh receipt, original physical source/manifest, complete log and upload/log attestations; its full parsed output equals the worker's frozen result. Physical upload370480 LF bytes, SHA256 `5f10e2d73571d0cafa6d2fd4ab3dcf068f99979cf3d8c77470b63bab31fa0f58`; canonical reader `b1422b5cb05505b094c309c6fb4c11e5d9f7103de3a1e735bf9b632d3cca0ffa`. Server content identifier `c33c8cf593da5d716c78da21c446a474088943e34a3085e35d49d5d2155b6e42` remains separate from file SHA.

H200 official5tests/6benchmarks/6profiles and complete private54cases/378observations/54restorations/216captures/1152timed replays/cleanup pass. Preparation218.246490452s fits295s. Thus this is a valid complete experiment, not another setup failure. All24 crossed component performance factors fail; all6 affected T1 cells regress. The24 T1 control/candidate factors range0.670398..0.738631, implying35.4%..49.2% higher candidate latency. Their descriptive geometric factor0.707998 gives **41.24% higher T1 component latency**. This is not a measured full-layer slowdown or an instruction-level causal attribution.

Official2.1658583028230054x versus SGLang is a different comparison and does not rescue the matched loss. Exact native `dbdd3381` remains PARKED with no repair, repeat, T8 expansion, tuning, layer/public/default action or follow-on allocation. See [worker terminal](../../../docs/steering/2026-11-02-worker-t1-moe-terminal.md) and [captain closure](2026-11-02-captain-t1-terminal-decision.md). Existing losing-run/ruled-out rows are already published; this report does not duplicate them.

## Required ROI screen for the next captain allocation

The current user instruction is to avoid spending another check on a tiny optimization. **Do not spend another check on this T1 version. Before a new target request, write a credible path to at least1% net six-load full-layer gain, and favor mechanisms with several-percent potential.** Readiness, source novelty and unused quota alone do not satisfy that screen. A1% standalone gain is insufficient justification.

Existing captain owns this decision; existing worker remains sole judge dispatcher. The allocation request must name actual selected loads/routes,46:16 and six-load weighting, net endpoint savings needed, source/profile/complete-implementation evidence for exposed savings, all extra work and integration costs, strongest competing task, uncertainty and the smallest distinguishing measurement. If only T1 is affected, that means a supported path to roughly4.6us net rather than counting total expert duration as removable. If no credible path exists, HOLD the target request and select a materially different mechanism; do not create more readiness or tuning work around this closed implementation.

Source correctness, BF16 boundaries, per-expert normalization, memory ordering, progress, replay/lifetime and official numerical gates remain mandatory. A later separately authorized component pass still requires complete-layer qualification; any invalid result or valid subthreshold/regression follows its frozen stop rule. This adds no target permission, changes no old acceptance statistic and preserves current ownership.

Publication reconciliation: newer captain **48111ace**,11:08:16AM, has already allocated an [isolated attention-continuity source task](2026-11-02-captain-attention-continuity-allocation.md) to the existing worker. It is source/CPU/fixed-secondary-compile only, with no target harness/GPU/check permission. Preserve that bounded task rather than creating a duplicate or freezing the roadmap to T1. Before any later check, it must satisfy the same new ROI screen using its actual full-context16/62 weight, affected four-load scope and segment-boundary exposure; total attention activity alone supplies no saving budget. The current next gate is the source task's concrete safe candidate or STOP, followed by captain's ROI-backed target decision. No completed T1 review is blocking it.

Fresh11:10:09AM state: **MoE8 private/5 scored; layer0 private/5 scored; no live jobs**, overall/layer366, public incumbent unchanged. Capacity is not permission. No check, submit, native change or external worker message was issued by this audit. This repository handoff communicates the ROI priority; no acknowledgment is claimed.

Reproduction: run [analyze.py](../../../docs/steering/evidence/2026-11-02-t1-layer-roi/analyze.py) from repository root. It reads immutable evidence at5f687801 and writes [analysis.json](../../../docs/steering/evidence/2026-11-02-t1-layer-roi/analysis.json); [snapshot.json](../../../docs/steering/evidence/2026-11-02-t1-layer-roi/snapshot.json), fresh own receipt and independent reader output retain the verification evidence. A read-only quantitative subagent independently checked the sensitivity arithmetic; it created no implementation or dispatch work.
