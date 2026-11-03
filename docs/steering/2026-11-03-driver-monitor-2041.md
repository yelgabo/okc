# Driver monitor: fixed-cost calibration and the H200 parity plan

November 3, 2026; source **05cf5286**, fresh boards/jobs/quota **20:43:29 UTC**. The [current official page and its newly referenced asset](evidence/2026-11-03-driver-monitor-2041/schedule.json) still close kernels **November 5 at00:00America/Los_Angeles**. The asset changed to index-DsBuz7Qf.js, hash c2b75cb7; the deadline did not change. No root judge/status poll, duplicate collector, worker creation, SSH/GPU probe, public submission, pod/driver change or message occurred.

## Fixed-cost check: measured terms and estimates

Own **FC-01 24068e6fd73e4240**, source e12b40973e77418e774c283cd2533a4f689d1ab88ccb851f246b3d0db5d79a29,239038bytes, retains incumbent picks and kernel entry. PASS9tests/30benchmarks/30profiles, **3162.787500us/1.527386838x**, CI[1.504476036,1.550297641]. It is above publicC23161.918773us and creates no new numerical promotion lead. This is a valid diagnostic, not an improved callable. [Primary CLI receipt](../../muse_bf16_proj/experiments/profiles/fixed_cost.24068e6fd73e4240.json), [owning protocol and analysis](../runs.md).

The experiment used12rounds by5repetitions,rotated arm order and cold-L2 flushes. Raw preparation timings: empty graph2.91us,tiny kernel4.86us,paired node increment1.92us. The judge's separate calibration subtracts an overhead computed from one-node/two-node graphs. For its ten M1/M8 cells, **one minus overhead equals two minus one:0.992–1.152us**. Root independently derived these values from each retained replay_overhead record. These raw and adjusted terms must remain separate. [Reproducible calibration rows](evidence/2026-11-03-driver-monitor-2041/fixed-cost-review.json).

Null CTA-count changes add0.66/2.61us in the tested configurations; cluster8/16 adds about0.03us versus the comparable launch. Those are bounded observations, not universal free-cluster or CTA-cost laws. In actual streaming kernels dispatch can overlap other work.

**The simple streaming controls do not measure a bandwidth floor.** Real kernels beat them by2.11–2.77us; those controls do not pipeline loads across iterations. The4.58TB/s/intercept fit spans different real kernels. Residual ramp/tail estimates depend on assumed bandwidth and a subtraction model; they do not isolate removable time or establish an architectural ceiling. The earlier2–3us-above-byte-floor figure cannot simply be called removable launch overhead. Fusion can change node count, so it still requires its own complete-chain comparison; this calibration does not disprove fusion or justify multiplying a node estimate through the layer as a guaranteed saving.

All small-arm MADs<=0.10us,below the frozen0.30us repeat trigger. **Close FC-01 after its one valid check**. No new streaming-control check or unrelated tuning allocation follows from its analysis.

## New local H200 resource: preserve the existing operator and calibration gate

Lead91b6a456/ceea5c2b reports the H200SXM pod is reachable:132SMs,driver595.91.07,torch2.8.0+cu128,CUDA12.8toolkit. The user explicitly chose **continuous operation for about16hours**, superseding the earlier stop-between-bursts line. Root must not stop it between bursts or change its lifetime/helper. The existing box worker owns synchronization,runner construction,Nsight installation and parity publication. Root neither created the pod nor verifies remote state through an independent probe.

First allocated task: parity-check the **four current public files** against judge cells,with **within3% per cell** as the lead's timing criterion,then publish docs/box.md. No tracked docs/box.md or parity capsule is present in this inspected ref. This is a publication limit,not proof the worker is idle or blocked. A reachable H200 and successful builds do not establish representative timings or numerical qualification.

For meaningful parity, the existing box owner should record source/job identity,effective compiler flags,driver/library versions,cold-cache/graph method,actual prepared picks and layer T8edge masks,plus replay outputs. Different prepared dispatch is not source parity merely because bytes match; current pdld's public/private mask history remains relevant. These are measuring-instrument obligations,not a new Driver approval or reply gate. Workers can use published validated parity directly within their scopes.

Local paired measurements may resolve small changes beyond their duplicate-control noise. Three-percent cross-machine agreement alone does not prove the1.04% selected-twin lead, and local results remain labeled local rather than official scores/ranks. Fresh official qualification remains the release evidence. No root SSH session,second runner,collector,remote job or new paid-resource instruction is created here.

## Next owned work and ROI

**P1 selected-twin clean release preparation remains the concrete BF16 priority.** Existing TLB/XMCowner/lead should build the exact nine-choice clean dispatcher from00005928's3128.993210us point,preserve source/packing/numerical/order/progress/lifetime/fallback contracts,and reach one complete-call/all30-cell matched release decision under its separately explicit owning allocation. No clean artifact,source-bound qualification ID or pickup is newly published in this ref. This is a retained material numerical lead,not a proven prefetched-kernel gain; failed warm/XMC treatments remain closed. Source cleanup/compilation does not depend on pod timing parity or a quota reset. Validated local parity can later support relevant correctness/profiling/comparisons without competing with the box owner's initial calibration.

Layer pdl8e27b5 and MoE BCLM257751 remain their existing owners' next official checks after16:00Pacific; both judge quotas are actually exhausted. Preserve those slots,source identities and prospective gates. The local box can support their authorized implementation/validation after its calibration; this monitor allocates no new remote or judge experiment and does not preempt the box worker. Original MoE reminder stays paused.

Current publicf931T8six/private4a737four history remains exact. The new2c84diagnostic strongly supports merge-order variation in tested stand-in chains; it does not itself certify all full-layer inputs or rewrite the original conditional authority. Existing PDL owner retains its single source/authorization/repeatbase disposition and already-authorized pdl8e check. No repeated alert,withdrawal or new human question is created by this monitor.

## New Aravind source: distinct publication, still an unqualified comparator

[Scout05cf5286](2026-11-03-scout-aravind-moe-2022.md) pins **dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40**,91057-byte submission. It is genuinely newer complete source than the old399305/v6 screen:custom down/normalization paths,optional in-GEMM routing,and a guard pairing raw-global routing IDs with its matching custom normalization. Do not carry oldv6 findings onto every new path without reading its actual final selection.

The reported1.8696x/job366c4a635e5f4b01 has **no retained raw receipt or final prepared plan**; the author says new WGMMA variants were unselected and the small score movement is within noise. This establishes neither a new>=2us candidate nor a reason to displace the current own selected-twin/BCLM/PDL queue. It remains reviewable source with exact representation/precision/publication/progress and complete-call obligations. No foreign code was executed and no new research census or target allocation follows.

## Current state

[20:43 snapshot](evidence/2026-11-03-driver-monitor-2041/live.json): **overall/layer370,MoE354,BF16392,attention287,prefill662**. Public numerical winners unchanged:layerf93110374.616018us/2.180350575x,MoE28215737.988910us,BF163161.918773us,attention1446.190595us. Private/scored: **layer0/0,MoE0/2,BF167/2,attention10/2**,prefill untouched. Shared list has no live jobs at this instant; it can miss in-flightPOST/list lag and is not authority to dispatch. Different claims/ranks remain dated; do not add component or cross-job changes as a causal overall forecast.

All five boards refreshed; [promotion guard](evidence/2026-11-03-driver-monitor-2041/promotion-check-summary.json) passes **16lower private numerical leads,zero missing dispositions**. Existing BF16selected-twin clean-release task and PDLreview remain visible. No new public score or release decision occurred. Pod boot/parity preparation is not a scored improvement. Stay quiet on unchanged/running/source-only state; notify a meaningful measured result,changed disposition,real blocker or required human decision.
