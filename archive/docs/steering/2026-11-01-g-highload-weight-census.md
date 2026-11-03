# High-load G expert weight requests: retained capsule and local census

November1. No new kernel, H200 dispatch, public replay or submission by this worker.
This addresses T32/128/256 independently of the earlier T1/T8 zero-duplicate proof.

## Exact #303 source ownership

Source MOE_CUDA_SRC SHA256c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71,
pure stage file295ca590..., public startup-wrappedba9e327c.... B4FIN32 and main128/256
are WGS, SK_T8, FT_T64, SC6W1 on the measured native path. The same route prologue
builds one skinny job for1..8rows and ceil(count/64) fat jobs above8, each at most64rows.
`wgs_wproducer` uses S.jE[j] to address `(expert,256-output-row unit,256-K stage)`;
the job's token-row offset does not change its weight address. Each job consumes32W13
and16W2units,16stages each. A hot expert crossing64token rows therefore requests its
entire weight payload again for each extra job. Different CTAs also split output/K work,
but a single job's unit/stage ranges are disjoint; they do not duplicate within that job.

SC6 stage bytes35840 give per unique expert:
W13=32*16*35840=18,350,080B; W2=16*16*35840=9,175,040B; total27,525,120B.
These include packed code+block scales, excluding tiny global scales/norm, activation,
fixup and metadata traffic. Byte-scale layouts differ; do not reuse the tiny28,311,552B
figure here. B4FIN changes final aggregation, not this64row job/weight map.
T32 cannot cross64under unique-top8, hence no repeated group weight payload there.

## Exact retained H200 bank

Decoded WGF1 capsule from owning job `1c6ebc76b91d4b7f`, prescribed M256/seed4517,
input generator seed4517+256=4773. Header132CTAs/256tokens,1196u32 fields, raw decoded
SHA256 `6b47a5902f04041daf7cddeb0de6f46db2ef293ff39e9ae5d9823bf2311949e6`.
Token masks contain one bit per selected local expert; generated top8 has unique IDs,
so bit counts are the actual per-expert token/route counts. Writer-presence bits agree.

Counts expert0..31:
`[1,32,8,22,13,14,18,0,0,0,0,3,2,0,0,13,14,20,0,0,7,11,14,11,121,0,0,4,8,0,0,1]`.
337local routes,20active experts. Token-local-count histogram0:53,1:108,2:64,3:23,4:8.
Expert24 has121tokens: jobs64+57, so total21jobs versus20unique payloads.

|Phase|Mandatory unique payload|Requested payload|Extra|
|---|---:|---:|---:|
|W13|367,001,600B|385,351,680B|18,350,080B|
|W2|183,500,800B|192,675,840B|9,175,040B|
|Total|550,502,400B|578,027,520B|27,525,120B|

Extra=4.761905% of logical requested bytes (5% over mandatory). This is measured input
population plus exact source arithmetic, **not measured HBM traffic or elapsed benefit**.
The old frontier timestamps were for final-read tail and do not attribute duplicated reads.

Other retained natural SC6CORE aggregates, job05e51ddeb7014dd9:
M32seeds901/902 maxcounts6/8, M128max25/26, M256seed901max62: zero extra64rowgroups.
M256seed902 has341routes,26active,max69; at least1and at most4experts exceed64
(the minimum26+5*64=346 would exceed341). Exact omitted vectors prevent choosing
between1..4extra jobs, hence extra27,525,120..110,100,480B and3.704..13.333% requestfraction.
Other frontier banks retain unique-expert counts but not per-expert token masks; no extra
counts were invented. Owned53c07507b9fd4f8c M32 duplication used8row jobs, not thisG64row
map; it does not establish the same G opportunity.

## Prespecified local census and fidelity

Root authorized local CUDAinput-only generation. Before execution, fixed normal/ep8/busiest,
M128/256, seeds65001..65032 inclusive (64banks), no MoE model weights. The exact public
reference.generate_input AST was executed with its full hidden-state RNG consumption,
Student-t popularity.03 and top8, busiest original global group rotated to0..31. Diagnostics
add only selected-group metadata after routing. Every original group0..7 appears in the sample.
Backend: PyTorch2.10.0+cu128, CUDA12.8, RTX5070, capability12.0. Reference LFhash and all32
counts per bank are in evidence/census.json. This is normal standalone routing, NOT actual
layer logits or the task's separate skewed distribution.

**Cross-H200 reproduction failed**: local M256seed4773 differs249/256 masks from the
retained H200 bank (local max76 vs retained121). Four SC6aggregate calibration banks also
differ. Root/backend/RNG/version differences may explain it, but cause was not established.
To separate source from backend fidelity, we executed archived `_gen_input` function
SHAa0ce4f06ed4e7fa78578298b9314a521df4b7b2f33dcaf7ecdad0658069e2412 from standalone
whole source795ac642... on the SAME local GPU: hidden_states,IDs,weights are all bitwise
equal to public reference at seed4773. Thus source mathematics/RNG consumption agree locally;
the new census is an independent distribution sample, not reconstructed H200 banks.

|M|Banks with extra groups|Mean requestfraction|Median|Max|Total extra jobs|max expert count|
|---|---:|---:|---:|---:|---:|---:|
|128|8/32|1.3300%|0|10%|8|128|
|256|24/32|5.0516%|3.9231%|12.5%|42|250|

No seed/path tuning or generated candidate uses these seeds. All records, selected groups,
calibrations and full masks for the retained capsule are committed as data evidence.

## Score sensitivity and decision

Current607e weighted layer/expert envelopes: V16=269.346774/161.183871us,
V32=359.298387/204.412903us. D32 and other four loads receive zero opportunity here.
As an explicitly hypothetical proportional model, remove each bank's duplicated request
fraction from its profiled expert duration and charge all of it to layer critical path.
Across32prespecified paired seeds this produces geometric factor1.006278, median1.004412,
range1.0..1.014759;8/32exceed1%. This model is **not** a physical upper bound or forecast:
compute stays mandatory, L2 may already serve repeated requests, stalls/overlap are nonlinear,
and standalone routing differs from #303. The retained capsule alone gives roughly0.46%
whole-score sensitivity under the analogous V32-only proportional assumption.

Logical duplication exists and is common in this local M256 sample, but current evidence
does not justify a new high-confidence >=1% layer candidate. Park a rewrite/extra judge check.
The key missing observation is DRAM versus L2 traffic for repeated expert groups under the
real WGS schedule, plus exposed weight-producer stalls. First gather hardware-counter/trace
attribution on a fixed independent H200 bank if that capability becomes available; do not use
byte counts as throughput or add them to the already closed final-reader budget.

A simple FT_T128 knob is not portable: WGS_NT8→16 doubles each X slot32768→65536B;
with3slots it adds98,304B to the observed231,168B, totaling329,472B. Hopper's per-block
limit is227KiB, so that exact geometry cannot fit. It also needs new NTT16MMA specialization,
doubled accumulators/fixup ABI, not an unreviewed flag. NVIDIA primarysource:
[CUDA12.8 Hopper guide, occupancy](https://docs.nvidia.com/cuda/archive/12.8.0/hopper-tuning-guide/index.html#occupancy).

A materially different residency-compatible strategy would multicast the same stage to
2cluster CTAs owning different token groups, retaining separate accumulators and existing
per-CTA footprint. That needs matched expert/unit schedules, cross-CTA all-reader retirement,
cooperative cluster admission and measured benefit beyond its synchronization/imbalance costs.
It is not implemented/approved by this census. Existing L2 sharing can already reduce its
benefit. Reopen only after measured duplicate DRAM/exposed stalls support >=1% whole-score
across independent banks, then require fixed four-arm/three-seed >1% and2xnull, fresh/skew/remote
correctness, no residency loss. No donor kernel copy or fence relaxation.

## Reproduction and tests

`tools/dl_g_next_expert_traffic.py` decodes capsules/models bytes and performs the fixed
local census; `tools/dl_g_next_expert_traffic_calibrate.py` compares the archived generator
on the same backend. Local interpreter `/root/.cache/okc-graph-cu128/bin/python`.
Evidence `docs/steering/evidence/2026-11-01-g-highload-traffic/{census,summary}.json`.
Three CPUtests pass: exact retained counts/phase bytes,64row boundaries, malformedcapsules.

`python -m unittest discover -s tools -p test_dl_g_next_expert_traffic.py -v`

No kernel changes, GPUjudge quota or public actions. Parent owns index/publishing.
