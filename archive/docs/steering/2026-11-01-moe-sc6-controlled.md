# Standalone MoE lossless-scale causal comparison

Registered before dispatch, November 1, 2026; parent evidence `8d34956`.
One MoE-board check authorized. No layer/prefill quota, public submit or donor
production adoption is authorized. The coordinator owns index/publication.

## Opportunity and exact scope

The new comparator's exact check `b65c4c878014416c` passed5 at2.395702934x
(95%CI2.359767390–2.431638478), versus published owned47 `abe7651dbaa44458`
at2.215712x. Their1.081234 ratio is cross-run and cannot causally attribute8.1%.
The discrepancy remains open. Baseline profiler `eae9a6b7d5dd4809` and
`tools/hotspots.py` on the comparator show the native fused kernel is its whole
GPU path; SGLang's dominant work is Marlin. This is not evidence of HBM saturation.

Selected mechanism: losslessly encode eight block-scale bytes per thread into
six bytes. Per WGS stage the code bytes stay32768, while scale bytes change4096
to3072; total36864 to35840 (2.7778% fewer logical requested weight bytes).
This applies to32/128/256;1/8/16 are unchanged negative performance controls.
The ideal exclusively byte-limited stage ratio is36/35. Applying it to exactly
three of six equally weighted loads gives `sqrt(36/35)=1.014185`; this is a
conditional byte-model ceiling, NOT a measured speedup or universal bound (decode,
cache, register and scheduling costs can differ). A result substantially above
that would need investigation rather than attribution solely to reduced HBM.

This experiment uses actual supplied256-expert weights and real global IDs0..255.
Only the local0..31 weight shard is packed, as required by the official EP8 task;
input global IDs are passed through to the native router without remapping them
into an artificial32-expert input universe. Four arms share read-only layouts but
have four private mutable scratch banks: byte, byte duplicate, SC6, SC6 duplicate.
Duplicates share the same compiled code per condition and test private-workspace
placement/null variation; they are not independently compiled implementations.

Pinned attributed source is keell0renz commit8e30e1be994db4310794b8d04e90d18d895deb84,
raw fileSHA2d931f09133d058053dda833b51f1867ea06cdcac0c912521ad37e8f7853185c,
LF-normalizedSHA795ac642adc3370460ff04166ae19b5aac172a48aba0170b94dac2a6a3430410.
The generated single file is private scratch and retains its notices. Only the
generator/harness/tests/receipts are committed, not external implementation text.

## Conformance common to every arm

Both conditions use the previously reviewed owned safety transform: ordinary
same-stream cooperative launch, checked all-resident grid/SM90/grid<=136, NT
reader arrivals and proxy bridges, all three global xA/xB writer proxy fences,
acq_rel grid counter, MFINoff, sentinelDFoff, YBFoff, and empty-job scheduler guard.
Standalone main differs from the layer at four rather than eight WGS GEMM2
materialization sites; the shared patcher now accepts that exact expected count,
without changing its layer default. Its four existing layer tests still pass.

Each GEMM1 gate/up result rounds toBF16 before FP32SiLU; intermediate rounds to
BF16; GEMM2 result rounds toBF16 before gamma. Actual SGLang baseline materializes
BF16(1+raw BF16gamma), and this pair does too. The FP32 official mathematical
reference instead uses FP32(1+gamma); do not conflate those two sources. Official
correctness remains required. TF32 is disabled once in serial setup, before either
arm. No concurrent GPU preparation or tuner is used. The dedicated M8 issuer is
omitted from both arms; this is a causal SC6 comparison within a repaired backend,
not an exact reproduction of the external bundled dispatch.

SC6 admission requires both the actual local positive-normal PRMT check and the
0x40..0x7f byte-domain check, so unsupported weights fail closed for this diagnostic.
There is no lossy quantization, activation packing, input-derived cache or dropped
ordering operation. Different registers from the decoder are part of the tested
mechanism; spills must be zero and dynamic/shared residency equal before timing.

## Preregistered validation and measurement

Six loads1/8/16/32/128/256; two fixed held-out seeds901/902, routed through the exact
standalone normal/heavy-tailed/busiest-shard distribution (no seed search).
For each input all four arms compare against an independently packed SGLang Marlin
baseline using privately cloned local raw weights, at official tolerances global
relativeL2<=.015 and max row<=.075 with official0.1 mean-row norm floor.
This is an independent backend control, not the FP32 mathematical reference.
The official judged byte arm still receives the real task's mathematical checks.

Before timing, the same four captured graphs receive fresh x/global IDs/routing
weights by pointer-preserving copies and16 queued replays, then all outputs are
checked against a fresh baseline result. Real global IDs32..39 form the empty-local
case; outputs are prefilled17 and must become finite exact BF16zero. Original data
are restored and outputs rechecked. After20 cold timing replays every output is
checked again. Both condition resources, occupancy, local bytes and local route
counts are logged. Timing flushes256MiB before each arm, interleaves arms and
reverses arm order on alternate repetitions. No prepare-time fastest-arm policy.

Acceptance for an owned compression implementation follow-up: BOTH duplicate
paired geometric six-load ratios >=1.01 and gain >2x the absolute-log null from
both within-condition duplicate pairs; every case regression<=max(.5us,1%byte).
All cases, official correctness, fresh replays, empty outputs and resource gates
must pass. Failing any gate precludes promotion. This check remains measurement
only even if it wins. The280s between-case guard includes initial prepare time,
does not interrupt nvcc, and does not establish a hard prepare deadline.

## CPU gate before dispatch

Two source/actual-summary tests pass, including acceptance, neutral, loss and
compressed-duplicate-null failure branches. Four Torch2.10CPU tests pass:
actual pinned encoder versus instruction-equivalent CUDA decoder on4608 thread
tuples (including all64 values in each field), rejection outside lossless domain,
actual WGS packing with bit-identical code stream, and execution of the complete
12-case/four-arm host harness under a graph/resource/clock facade with real CPU
tensors. A deliberately stale captured-x implementation fails the fresh gate.
These are host/encoding proofs; no CUDA compilation or H200 measurement yet.

## Next action after result

If compression wins the registered whole-score gate, design its owned integration
with the same byte-domain fallback and exact decoder tests, then independently
prove its ring release counts and physical slot layout. Do not copy the private
comparator into production. If SC6 is small/neutral/losing, park it as the main
explanation of the8.1% bundled discrepancy on this repaired backend. The separate
per-expert producer/owner/final-read frontier probe is then the strongest causal
next question at128/256; no unsafe MFIN flag should be enabled to get that answer.

## Terminal H200 result: compression-first gate fails

Actual MoE check `05e51ddeb7014dd9`, source commit7886294, frozen file375858B,
SHA256 `54d819e6482aaaa16c8a4e6d38f60e1e98b044b8888120f40b5ab7f3bccfb4cf`.
Receipt verifies board=muse_moe_w4a16, H200, mode=check, status=done, passed=true.
All5 official tests pass. Official byte-arm score2.259310880x,
95%CI2.225421217–2.293200543; prepare116.574s. This private attributed diagnostic
is not a production candidate, and its cross-run score does not establish an owned
gain or isolate any one repair against the earlier bundled comparator.

All48 initial,48 fresh-pointer/replay and48 empty-local computing-arm comparisons
pass. Worst independent-Marlin error is global.006205786, row.006743266. Official
FP32-reference tests (including M4/33/257, skew128 and M1) report global errors
.00447–.00461. No threshold was relaxed. Empty probes use real global IDs32..39,
prefilled output17, and exact finite BF16zero. Replayed-output gates occur before
timing and after it, not on a stale pre-replay capture result.

| M | byte mean of 4 case/arm medians, us | SC6 mean of 4 case/arm medians, us | paired ratio A | paired ratio B |
|---:|---:|---:|---:|---:|
|1|35.248|35.184|1.001122|1.002923|
|8|81.504|81.168|1.007443|1.001369|
|16|91.360|91.300|0.995824|1.005487|
|32|129.412|127.276|1.005924|1.027565|
|128|183.776|183.208|0.998603|1.007684|
|256|223.776|225.120|0.990032|0.997959|

Each paired column geometrically averages the two seeds within its fixed duplicate
pair. Do not select the better duplicate or M32 after seeing results. Across all
six loads the registered paired ratios are **0.999806837 and1.007118932**. The
duplicate absolute-log null is.006403393 (0.64034%; twice=1.28068%), and the
case regression gate also fails: at256/seed901 the first SC6 arm is2.576us slower,
above its2.112us one-percent allowance. Therefore the preregistered gate is FALSE.
The balanced duplicate ratio1.003456225 is below null; it is not a repeatable win.

Both main conditions have identical false/true vectors:96registers,0local bytes,
0static shared,231168dynamic shared,1residentCTA/SM,576threads. Tiny/Tiny2 have
149registers,0local/static,214784dynamic,1residentCTA,288threads; MID has152registers
with the same remaining vector. These are function/occupancy-query resources,
not measured SM utilization or HBM/L2 throughput. The grid is set from actual
device SM count and launch-guarded, but the numeric count was not printed; no
observed numeric grid claim is made.

Official byte-arm per-load median latencies (different fresh official inputs from
the paired fixture) are26.133/65.000/93.093/136.053/189.221/231.217us at
M1/8/16/32/128/256. Preserve their distinction from the paired means above.

Fresh board snapshot before dispatch: owned rank377 at2.2157x, rank50 at2.4534x
(`48494e439a5749ff`), rank10 at2.6077x (`1084b630433041f9`). Exact owned receipt
2.215712 still needs about10.727% more score to the displayed rank50 value.
The exact historical combined external check2.395702934 remains about2.408%
below that threshold. Neither this repair nor SC6 establishes top50 feasibility.

Full receipts/log/profile/normalized summary:
`muse_moe_w4a16/profiles/sc6_compare.05e51ddeb7014dd9.*`;
dispatch and board files have the same sc6_compare prefix. Independent today_proj
review cleared the exact frozen file and independently reran2source+4TorchCPU
tests before dispatch. No public submit occurred; production sources unchanged.

**Decision:** close compression-first as the explanation of the8.1% comparator
discrepancy on this fully fenced/BF16 controlled backend. Do not implement or tune
an owned compression port on this evidence, and do not spend another check on a
selective M32 policy. This does not rule out compression with a materially different
decoder/layout, cache behavior, or later low-cost combination; reopening requires
new measured mechanism, not a tile/flag sweep.

The smallest score-relevant engineering next action is the separately reviewed
per-expert producer/owner/final-read frontier diagnostic on true standalone128/256
inputs, with all synchronization retained. It must quantify its own instrumentation
overhead and treat pre-epilogue writer samples as optimistic issue frontiers, not
publication times. Only if the fixed-owner critical-path ceiling is meaningful
should generation-safe readiness counters/finalization be implemented in owned
code. Old bounded-poll MFIN remains prohibited; this result does not rehabilitate
it. Existing owned published47 remains the submission recommendation for MoE.

## Narrow external discussion context (read-only, not our measurements)

After the result, root supplied authenticated research thread
`t_mvqqjkzx783052c6`; no thread body or code is checked in. Post
`p_mvt72bdp3eca20d5` (alygg77, October29) attributes its WGS unit-end cost largely
to consumer work: roughly1.8–2.0us whole,2.2–2.6us split,3.6–4.4us finisher,
versus~.2us slot waiting. Its helper/producer handoff experiments reportedly lost
9–15us. These are their architecture, instrumentation and runs, not evidence our
same boundaries have those costs. Their phase-B MID consumer measurements do not
establish an exposed final-normalization opportunity or authorize fence removal.

Post `p_mvslupf3cb173142` reports a different five-bit codec losing despite fewer
bytes, with spills and decode work implicated. Our SC6 pair has no spills and
still fails, so that post does not establish our causal explanation. Scale-sharing
posts `p_mvso7nv4fa0723b3` and `p_mvswpjvja89ae1e7` concern K-permutation/paired
decode, already present in the external source. Owned76/78 tests64254f49676c4ce6
andc8fc5f669fa34d06 limit that existing alternative; the posts alone do not reopen
it. They reinforce measuring the producer-to-final-owner critical path before
assuming global rendezvous time is removable.

## Subsequent source review, no second job by this owner

Root's separate frontier generator was independently read-reviewed after the
SC6 result. Current generated SHA at review:
`76c5b0e7483aba481f605a4d2a93363e1e8e968288f0bf977660da115ff9628b`.
The five-slot scratch ABI uses a zero-length unused fourth debug placeholder and
an extent-checked fifth trace tensor. Writer locations are CTA/warp/expert private;
the issue stamps precede PB epilogue stores and are not publication events.
Macro-off code strips to the pinned original device body. Four independent CPU
model/source/capsule tests pass. Three host review findings were repaired before
clearance: assert the resource gate, check timed outputs before fresh-input copies,
and use unique nonlocal global IDs32..39 for the empty case. No source-level
blocker remains for root's diagnostic; no GPU correctness claim follows. The trace
representative is the median of five saving bounds, not the median-latency trace.

Root also proposed a simpler future publication design: keep original MFINoff,
separately add complete per-expert publications, and defer the existing cooperative
grid rendezvous/generation advance/cleanup until after every final read. This can
avoid a new epoch scheme, because CE cleanup follows that retained rendezvous and
the next ordinary same-stream call waits for completion. It still requires exact
whole/split writer counts, participating-writer publication, acquire-before-read,
the existing own-CTA join before shared-ring reuse, and no bounded poll fall-through.
The relevant formal chain is in [PTX8.7 sections8.9.2–8.9.5](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-synchronization).
No implementation is authorized before the measured frontier envelope clears;
publication-only controls must price its overhead rather than compare solely to
an unsafe flag-enabled variant.
