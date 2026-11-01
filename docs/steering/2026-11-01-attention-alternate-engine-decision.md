# Attention architecture decision: alternate verify engine

November1,2026. Read-only source audit after6d351f1; no public submit, prefill
work or GPU experiment in this checkpoint. Owned production evidence pin
6da000f, known checked late-full candidatecbee36a...b356. Public alternatives
are measurement/learning only; no donor CUDA is committed or adopted.

## Meaningful target and scoring bound

Saved official board snapshot `leaderboard.2026-10-31-late-campaign.json`:
rank50 concept-guest2.7093, jobbd59d00e5c534f60; ours2.60429. Closing this
snapshot needs4.0322% score or3.8759% uniform latency reduction. This is not
a live cutoff. Calibration `scored.8914751e3b9f468b.json` supplies corrected
per-shape medians and46sliding/16full weights over six equally weighted loads.
Improving only v16full would need28.3967us reduction from101.555us tobridge
that saved rank50 gap. A10% v16full reduction yields1.3172% aggregate;10%
v1sliding yields0.9789%. Thus both routed rows could matter, but neither
source appearance nor a public wholeboard score establishes these gains.

## Official reference/backend alternative is already measured

`muse_attention_decode/baseline.py` pins SGLang a55ab7d5... and invokes
FusedQKNorm, FP8 cache store/query cast, FA3 paged causal attention, RMSNorm
gate. `submission.py:_fused_prep`303ff already fuses preparation and cache
write around FA3 with a final fused gate. Receipt43ef868a181e49b2 measured
this architecture1.128x, whereas current owned path is around2.6x. Full
FA3 main was94–97% of largebatch time in baseline2b3c25820fee4b2d; measured
page-size1 gathering was the limitation. Generic prep+sameFA3 is therefore
not an untested major improvement. Historical fused/ws paths are preserved
in submission.py and `tools/attn_make_bench_one.py` can compare `_one`,
`_fused`, `_ws`, `_fused_prep` without importing judge reference.

FlashInfer is a distinct potential backend, but its current public API is
not proof of installed judge capability. Official documentation exposes
paged NHD tuple caches, sliding windows, FP8 types, and planning wrappers:
https://docs.flashinfer.ai/api/attention.html
The official layout reference is
https://github.com/flashinfer-ai/flashinfer/blob/main/docs/tutorials/kv_layout.rst
Runtime lengths/page tables are fresh GPU inputs here; a CPU plan from their
values cannot be cached or host-synchronized inside the timed CUDA graph.
Version/ABI, GPU metadata rebuilding cost and permitted graph setup would
need investigation before a backend experiment. No guessed FlashInfer API
was implemented. Continuous-length mixing must likewise derive work from
current GPU lengths, not branch on host copies or reuse prior input metadata.

## Newly pinned split-path comparator

Root acquired public keell0renz/okc source at commit
`8e30e1be994db4310794b8d04e90d18d895deb84`, pinned rawfile:
https://raw.githubusercontent.com/keell0renz/okc/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_attention_decode.py
Local scratch `.build/today/kel/kernels/muse_attention_decode.py`, SHA256
`496d460e0bb1e7489684de2742f1ff22f3959d9d5fe30674199e74b3d002d651`.
Repository origin/path should be verified by root against its fetch receipt;
the exact local bytes above are what this audit read.

The source is explicitly labeled measurement-only at3237. It embeds
ZhimaoL/sglang Apache2 verify engines. Important default routing3249ff and
`_zh_route`5778ff: WST full verify B16..24 (therefore scoredB16, NOTB32),
VP2 slidingB1; VFG slidingFalse, VFGfullB1False. Otherrows retain vjeux path.
Do not attribute whole-file gains to an alternate engine on all scoredrows.

WST is materially different from our mode51:128-keytiles, one residentCTA/SM
(`wst_ct14b`3592ff), balanced streamK over requests, FP32fold every16tiles,
QK(next)/PV(previous) overlap and lazy rescaling. It changes work granularity
and engine layout rather than retuning the rejected stage ring. VFG uses a
separate per-request split engine and merge; VP2 specializes batch1sliding.
Our mode51 currently uses64-keytiles/two residentCTAs and fixed-reference
FP32 accumulation. These architectural differences deserve measurement,
not a claimed speedup or immediate source rewrite.

Critical confound: build flags5710/5722 enable `-DZH_SCH=1`, selecting a
piecewise-linear exponential and alternate packing; comments call the file
measurement-only. There is an exact-exp branch (`ZH_SCH=0`). A measured gain
in the default approximate branch cannot establish a conforming owned
architecture win or justify copying its math. Preserve all original results
and distinguish exact/approx paths. Also audit all writer/reader proxy
fences and mbarrier arrival ownership before any owned implementation;
public passing results do not prove its ordering.

## Exact missing comparator and prospective gate

The next justified off-board experiment is current owned checked source vs
this pinned public architecture with `ZH_SCH=0`, with original1 diagnostic
only to measure the math confound. Keep independent prepared modules/states
and identical fresh inputs; report actual route per row, both independent
nulls, official cache/math checks and resources. Initially target default
routedv16full/v1sliding with all other rows held as controls. A forced VFG
sliding16/32 route is a separate architecture question, not proof from its
currently disabled dispatch. No candidate promotion from this comparison.

Only if the exact engine saves a meaningful calibrated share of the rank50
gap beyond nulls should we implement an OWNED128-key/oneCTA engine or VP2
style layout. Require an independently understood resource/lifetime/order
proof and a concrete multi-row score model first. Otherwise close this
comparator and preserve the current checked submission. The prior mode51
counter-rank weighting7631033a0c93485d/4ff339332fef4aff, deepermode49ring
42c4739cf6b247c4, per-request carryc77ce71c81af4fb1 and currently-hoisted
PF3pageIDs are not the proposed mechanism and are not reopened.

## Prospective exact split-engine comparator gate

Authorized one off-board check, measurement only. Generator
`tools/attn_make_exact_split_comparator.py` embeds pinned scratch donor with
ONLY SCH0 and host resource-query hooks; production source unchanged. Native
routes only, no forced disabled route and no approximate SCH1 benchmarking.
Four independent arms ctrl/ctrl2/pubkel/pubkel2,19 ordinary math cases plus
fresh additional v16full/v1sliding banks, every arm checked against TWO own
FP32 references, official global/row/cache bounds and incumbent relative2%.
Failed correctness blocks timing. Resource gates reject local-memory spill
in selected WST/VP2, require WSTresident>=1 and VP2activeclusters>=2; own
resource metadata printed. No new actualresource claim before receipt.
Private donor buffers and independent module naming are retained.100cold
paired rounds, two input banks, seed311151; all12shapes preserve controls.

Preregistered meaningful architecture threshold: holding every non-native
row atctrl, pooled sixload improvement must exceed BOTH2xindependentlognulls
AND1.0% aggregate (roughly one quarter of saved4.03%rank50scoregap). Both
independent public arms must improve native affected loads on both held-out
inputs, with no selectedrow loss greater thanmax(null,1%). Smaller resolved
wins are recorded but do not justify a major rewrite. Passing is comparator
motivation only; owned implementation, all-input queuedstress and official
candidate checks remain mandatory before any promotion. Source, precision
and public-versus-owned attribution stay explicit.

Dispatched `4862f8e0dc8a4b5a` (muse_attention_decode confirmed), source3842beb,
generated SHA256b76e1ce7591b47dc46e721262f305567505c43e17cd1415d6e17e7581ec33402.
WST host resource queries are inside one-time reporting guard, as VP2; warmed
capture retains original host launch path. Offline AST double-parse confirms
bothSCH0 flags, independently rewritten duplicate extension names, proper
CUDA printf backslash-n, and exactly one TF32-isolated/restoring reference
wrapper. Private generated artifact lives `.build/today-attn/exact_split.py`.
Pending result; no actual counter/resource/performance claim yet.
