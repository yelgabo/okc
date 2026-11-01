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
