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

## Terminal exact comparator4862f8e0dc8a4b5a: matrix loses, VP2 isolated signal

Result passed16 official control tests/checks.ok/flags[], unchanged-control
score2.528391542x (95%CI2.490465669–2.566317415). All four independent arms
passed21/21 own two-reference FP32/global/row/cache checks. This is not a
public candidate check or all-input queuedstress. Full receipt/log:
`muse_attention_decode/profiles/exact_split.4862f8e0dc8a4b5a.{json,log}`.

|native selected row|ctrl|ctrl2|public exact1|public exact2|per-input public1−ctrl|
|---|---:|---:|---:|---:|---|
|verify1sliding VP2|11.10|11.10|10.34|10.37|−0.93/−0.70us|
|verify16full WST|114.02|113.95|126.11|126.40|+10.14/+14.21us|

Raw cold graph medians in microseconds include replay overhead. Independent
public2 signs match both rows/inputs. WST exact-math engine is substantially
slower; the registered native-matrix gate FAILS. Holding every non-native row
fixed and subtracting this receipt's official per-shape overhead gives
sensitivity public1 .997479993,public2 .996720914,ctrl2 1.000081757; pooled
cross .997059624 (−0.294%). Publicnull ratio .999239003. These sensitivities
use rounded AB medians and borrowed calibration, NOT candidate scores.
Other public vjeux-derived paths moved too; their gains cannot be attributed
to the new WST/VP2 architecture or credited to this registered native gate.

VP2 alone, a NEW restricted follow-up decision rather than retroactive gate
promotion, has calibrated ratios1.011088375/1.010634087; pooled1.010861205,
publicnull .999550694. Thus the batch1sliding cluster architecture plausibly
covers about one quarter of the saved4.03%rank50 score gap, while WST should
not be rewritten on this evidence. Needs fresh restricted preregistration,
source/layout/order proof and OWNED implementation before further benchmark.
No donor adoption is authorized by this result.

Resource hooks were required before native launches and all21checks completed,
so minimum WSTresident/local0 and VP2cluster/local0 gates executed successfully.
However their numeric printf lines were truncated from the retained log;
DO NOT claim exact registers, occupancy counts or shared bytes from this
receipt. Missing terminal resource summaries are an instrumentation gap, not
an excuse to infer values from source. Private module/state ownership was
built into the generated harness, but public workspace cleanup was not
independently scanned; all-inputqueued replay correctness remains outstanding.

Next justified source investigation is VP2's one-KV-head128-keytile and
16-CTA cluster-local reduction/exit lifetime, compared to our two-KV-head64-
keytile/global cooperative finalize. Do not transplant its source. Establish
whether this reduces duplicate prologue/finalization or changes critical-path
work and prove DSMEM writer/reader lifetime through cluster exit. Keep every
fence/barrier. No new check or forcedVFGroute was dispatched after this result.

### Owned restricted cluster merge: prospective gate and proof

Evidence screen4862f8e0dc8a4b5a's VP2-only signal; new OWNED source generator
`tools/attn_cluster_merge_patch.py`, no public source adoption. This hypothesis
changes ownership/merge, not query precision or tile math: current mode3 PRQ31
64-key/both-KV-head CTA,32 balanced tile partitions, two16-CTA clusters. It
preserves existing global cooperative barrier/parity/finalization, reduces
per-row global partial additions32->2. It is not a claim to reproduce VP2's
one-head128-key pipeline. Old418d8d492c074b88's q8 cluster engine used slow
mma.sync math and serial128 DSM reads; this specific modern wgmma merge has
not been measured. No generic ring/stage/reweight sweep.

Uniform native guard: L8,MODE3,grid32x1,window2048,nc1. Other rows call PRIMARY
unmodified control, so no incidental recompilation credit. Quotient partition
covers every64-key tile exactly once including short-context empty partitions;
empty CTAs do NOT take the former early exit, still initialize zero accumulators,
join BOTHcluster syncs and existing global grid barrier. nact32 gives all128
finalizer rows one owner. Copy=split%nc=0; b0 makes shared rowmod128 and global
row0..127 exact. The optimization is disabled ifnc!=1; no cross-copy aliasing.

End-of-loop own wgmma wait0 remains; newcp_wait0 andcsync join all pending raw
copies/readers before alias. DeadrawKV[0:49152] is reused as128x68float partials
[0:34816]; Q at49152 and later constants/epoch/flags remain intact. Every partial
dim and row sum has exactly one writer (CPUbijection). Allthreads CTAjoin then
cluster.sync, rank0 reads each16 peer partials and emits existing fixed-frame
FP32reds; secondcluster.sync ensures all remote reads finish before anyblock
exits/reuses shared storage. Existing fences/acquires/releases remain unchanged.
NVIDIA DSM lifecycle requires synchronization before peer access and before
block exit: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory

Host cluster opt-in cudaFuncSetAttribute is checked once during prepare.
First actual clustered cooperative launch uses its EXACTcfg to require
cudaOccupancyMaxActiveClusters>=2 before launch; subsequent captures don't
query occupancy. Resource export records regs/local/dynamicSMem/blocksSM/
activeclusters, resource gate requires local0, sameSMem and no occupancy loss
versus base8mode3, >=2clusters; numeric metadata repeats atterminaltail.

Four independent ctrl/ctrl2/cm1/cm2 builds/states.120cold paired rounds over
THREE timing inputs, all12 scored shapes plus test shapes. All arms use own
FP32/global/row/cache checks; candidate failures prevent timing. Additional
fresh gate queues64 distinct QKV inputs through all4 capture graphs (256
checked outputs), every output versus BOTHown FP32query references and official
cache/global/row bounds, no inter-replay hostsync, oneverdict drain. Three
negative controls (zero,stale,+10) mustreject; finalworkspace scans require
allbuffers/tickets zero. This is64freshcases/arm, not2000mathchecked replays.

CPU partition/bijection tests2/2pass. Actual CPU Torch reference-layout test
executes generated vectorreference math against independent per-token/head
attention; BF16queryrelative0, FP8query5.18e-6, below1e-3layoutcrosscheck.
These do not certify GPU lowering/resources; one authorized check remains
pending independent source/ordering review.

Preregistered restricted gate: onlyv1sliding rows allowed to earn score;
all others heldfixed. Corrected sixload pooled cross gain >=1.0% AND >2times
max independent lognull; all12 crosses (2candidates×2controls×3inputs) faster,
no unresolved selected-shape regressions. Missing resource/math/fresh data is
failure. If no repeatable gain, revert/close this ownership variant and keep
purecheckedbaseline; no public submit. Passing motivates candidate-onlyofficial
and largerfreshvalidation later, not immediate promotion.

### Independent ordering correction before dispatch

Coordinator review found a blocker in the initial key-count barrier argument:
an entirely empty cluster could finish later than the nonempty cluster's total
key arrival, then issue zero global reds after finalization or load a newer
epoch. Initial claim that key-count release was sufficient is superseded.
For clmerge ONLY, existing coop_barrier now receives weight1,total32; allother
paths retain hi-lo/n_keys. Every CTA must read its stagedepoch BEFORE its
arrival, so total32 cannot advance until all32 have captured the old epoch
and both cluster leaders' reds precede arrivals through unchanged CTA/cluster
barriers. New adversarial CPU test runs emptycluster0 last after nonempty
cluster1: oldkeycount reaches8 at16arrivals; newcount remains16/32 and cannot
release. All captured epochs stay0 until final32ndarrival. CPUcontracts now
3/3PASS. This correction was made before any GPU dispatch.
