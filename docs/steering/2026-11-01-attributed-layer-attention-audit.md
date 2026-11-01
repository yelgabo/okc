# Attributed layer fork: attention safety audit

November1,2026; read-only, no GPU or kernel changes. This audits the LAYER file,
not the separately augmented standalone attention. Pinned public commit
8e30e1be994db4310794b8d04e90d18d895deb84, root scratch
`.build/today/kel/kernels/muse_decode_layer.py`, SHA256
1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a.
Line anchors refer to these exact bytes. Root owns attributed fork and publication.

## Actual selected attention and precision

The layer embeds ATT7_CUDA_SRC122ff and `_launch7`11318ff, not the standalone
Zhimao WST/VFG/VP2 additions. `prepare`11263 sets att_ok=True. Effective base
selection11473 is att_ok && supported7 && fits7, with no actual q_len1-only
restriction despite stale comments11468. Later wrapper12533 also selects
this same engine for pinned QKV plans at128/256 tokens. All supported q_len8
shapes therefore can use embedded attention. RuntimeError fallback11488/12541
is not a correctness gate and must not be relied on after device execution.

`_att7`3727–3738 compiles QDIRECT0,PK2=0,W0ENABLE1 and architecture constants.
Source defaults PV8=1,PV8FLAT=1,Q8RAW=1 (164–171), EXPPOLY0,EXH8=0(249–250),
XP_NOFENCE0(260). No ZH_SCH=1 or standalone piecewise-exponential flag is
present in this layer path. `ex2`402 uses standard PTX ex2.approx.ftz.f32;
this should not be confused with the standalone Schraudolph approximation.

Scale chain: `_launch7`11359 passes sm_scale*qk_scale_factor*LOG2E.
Kernel1658 reads runtime kscale/vscale;1659 forms sl2=scale_log2*kscale.
Q normalization/RoPE precedes staging. Q8RAW makes qsc=1 at1743 for L8SM90,
so quantization is of unit-RMS query, not query multiplied by small kscale.
QDIRECT0 takes FP16 intermediate then E4M3 conversion1985–2014. Scores
multiply ssc=sl2 at2022; values use vscale at final normalization942/3021.
New K/V use reciprocal runtime scales1800ff and E4M3 conversion/write1808–1827.
This source chain is plausible for kv_scale1/64; it does NOT prove correctness
of every compile-selected branch or settle the stale comment about that test.
The FP16 intermediate introduces a distinct rounding boundary needing tests.

New-cache writes are limited to position in this split's [start,end),1820ff;
all new rows also staged in shared. New-key patch proxy fence1840, query
generic-store→wgmma fence2014 and Vtranspose fence2279 are present. QDIRECT
fence1763 exists but branch is compiled off. PK2=0 disables LSYNC fast bypass
2068. Do not remove these fences or enable timing-only switches in the fork.

## Workspaces and recaptures

`_workspace7`3746ff caches guarded FP32 accumulators/sums/counters per DEVICE,
not per layer state. It grows when needed. `_cluster_bufs7`3865ff similarly
caches guarded buffers. `_guarded`11400ff stores the full allocation in global
_GUARDS, preserving backing storage even when a cache entry is replaced;
this helps prevent pointer expiry across a later recapture. It does not make
shared buffers safe for overlapping graphs/streams or independent prepares.
Two states on one device otherwise share the same workspace and counters.
Conservative fork: make attention scratch ownership per prepared state, keep
all captured allocations retained for each live graph, and prohibit concurrent
reuse of the same scratch. Do not free/replace a captured allocation without
retaining its backing storage and proving no outstanding replay references.

Reset paths exist: request counters2794/2817/2883ff, lastCTA workspace/reset
2903–2935, flat finalization2952ff and epochs2969ff. This read found reset
intent, not a complete memory-model proof for every sense/parity/empty-request
branch. Recapture must cover changing batch/length/page64/scales and mixed
empty segments, with device error checks and guard scans after replay.

## PDL ordering and conservative baseline

PDL=True,PDL_MAX_T128 at96–102; `_decode_attention7`11385 explicitly passes
_pdl(T), overriding module defaultOKC_PDL0. `_launch7` encodes this into tma
bit16 and C++3589ff launches programmatic serialization when appropriate.
The per-segment prologue1533 calls pdl_wait BEFORE first QKV cp.async reads,
then triggers dependents. Empty-segment finalizers also wait2952/3131.
This is the intended QKV producer→attention consumer ordering. Global-to-shared
cp.async completion is subsequently waited before shared reads; the separate
local generic→async wgmma fences noted above remain necessary.

This audit does NOT prove the entire cross-kernel PDL chain, especially the
following projection's TMA/async reads of attention generic stores, every
producer trigger placement, or unusual no-segment paths. No reader-side
fence.proxy.async.global was found in ATT7 source. Absence alone is not proof
of a violation; root must map the precise proxy operations and applicable
PTX causality before claiming soundness. Conservative initial fork sets
PDL=False across the affected chain; then reintroduces only independently
proved edges with identical math/fresh stress. Merely checking pdl=False at
one call while allowing other secondary launches is a weaker baseline.

## Minimal fallback points and required coverage

1. Initially route q_len8 && kv_scale!=1 through existing FA3 path by adding
   the same guard at BOTH base11473 and later pinned-plan12533 selections.
   Ensure st['fa3'] exists; otherwise fail closed or retain known-good owned
   attention. Do not silently substitute an untested Triton split fallback.
2. Disable PDL globally for first conservative fork; preserve every device
   fence/barrier. Separately own scratch per state before concurrency claims.
3. Keep supported/fits guards11441ff, including power-of-two page sizes and
   <4GB caches. Keep FP8 cache writes and official BF16/gate boundaries.
4. Treat selftest11422ff as allocation/guard warmup ONLY: it invokes kernels
   once on zero caches/unit scales and performs no independent FP32 output
   comparison; it does not cover non-unit scales, recapture or fresh replay.

Required before relaxing fallback: official full-layer tests including
kv_scale1/64; independent BF16-query/E4M3-query references and cache8% bounds;
row/global bounds from TASK; both sliding/full L8 at batch1/16/32 plus page64,
mixed lengths and short contexts; every queued fresh input checked; negative
stale/zero controls; different independently prepared states and recaptured
graphs; sentinels/workspace reset invariants. Any attributed fork score remains
unqualified until these tests and selected-path/resource evidence exist.
