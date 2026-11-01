# Owned skinny expert-band transpose — source/compile checkpoint

2026-11-01. No GPU dispatch, public action, decoder change, external kernel adoption,
or production source change. Candidate generator: `tools/moe_skinny_band.py::patch`.
Source-only result pending independent review and a separately reviewed harness.

## Reopening evidence and scope

Independent preregistered local RTX census (today_proj8de9194,64 banks at16/32 tokens)
finds duplicate8-row jobs in14/32 and29/32 banks, mean repeated logical request fractions
4.5378%/12.0499%. Conditional proportional whole-six-load scenario1.03073 is NOT HBM,
measured latency, or a forecast. It is new population evidence versus old53c07507b9fd4f8c
6/8-case count-screen failure. That old closure concerned two-CTA multicast pairing;
this experiment changes same-expert unit locality without multicast or fewer requests.
Earlier token permutation, decoder-only835253441a214a1c, testwait and ring variants are
different mechanisms; their losses remain closed. No such mechanism is combined here.

Root refreshed hotspots against SGLang eae9a6b7d5dd4809: public D16=95.426us,
floor52.519us/profile79.5us; D32=136.747us,floor70.755us/profile128.1us.
Common835 control=95.4/132.6us, profiles86.9/144.0us from a different profiling run.
Skinny is the entire profiled kernel. Floors are not attainable transpose bounds.

Pinned input is tracked owned submission LF SHA
`c7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7`.
Public ec565a7/b35d3217 remains untouched and is not claimed byte-identical.
The fair future comparison uses the835 common cooperative-admission and BF16 effective
gamma mechanism on BOTH arms. `patch(source,cooperative=True)` applies the common launch
transform exactly once before the sole skinny device transform. Harness treatment is
restricted to16/32; other four loads use separate private unchanged control states.

## Single fixed hypothesis

Original skinny stream is physical job-major:512A units,256B units per8-row job;
each36KiB unit contains its output rows' complete K4096 weight tile and scales.
Different jobs of the same expert issue matching weights a full job apart.
The fixed32-unit transpose changes only unit order inside an expert's contiguous jobs:

`local v -> job=(v/32)%J, rowunit=(v/(32*J))*32+v%32`.

Each expert's full virtual interval stays at the same position and has the same length.
Canonical physical `(job,rowunit)` is used consistently by weight issuer, compute warps,
and helper epilogues. All units still execute exactly once, including partial8-row jobs.
No ragged weight-stage cost or stream-K fixup exists in this skinny kernel.
32 units are1.125MiB of weight requests/job; matching logical requests move closer from
18MiB A/9MiB B. This is a virtual-order distance, not physical issue simultaneity or L2 reuse.
No weight loads are eliminated, and HBM/L2 behavior remains unmeasured.

Activation register reuse lasts each32-unit run. It cannot retain two jobs cheaply:
the existing `xr[8][2] uint4` is64 scalar registers/thread. Full-unit alternation was
rejected. For full8-row jobs, one reload is64KiB. Phase A uses read-only default loads;
phase B explicitly uses `__ldcg` and bypasses L1. Requested bytes are not DRAM bytes.
Independent exact132CTA census cost model reports mean extra activation requests:

|Tokens|A extra bytes|B extra bytes|A CTA-job credits old / new|
|---|---:|---:|---:|
|16|684544|286720|139.4375 /150.875|
|32|2585856|1163008|146.5 /184.9375|

The final column is a real cost risk: once/job/CTA does not mean unchanged grid-wide
credit count. New mapping changes which CTAs touch each job. There is no claimed speedup.

## Publication and progress argument

1. Routing helper warp produces histogram `cnt`, expert offsets `off`, and job metadata,
   then its existing warp synchronization precedes initial weight issues. These arrays
   remain unchanged through position_pass and all compute/helper execution. Mapper needs
   no `ent`/`pos` data before position_pass.
2. Every issuer/consumer/helper applies the same bijection. Virtual CTA ranges and q,
   ring slot `q%3`, full parity, partial-buffer q index, waits, arrivals and proxy fences
   are unchanged. The initial issues may cover a different physical unit, never a new q.
3. Naive j-change credits would fence every band; removed. For physical group-job i,
   clipped prefix `P_i(x)=floor(x/(32J))*32+clamp(x%(32J)-32i,0,32)` exactly counts its
   preceding units. This CTA's credit is `P_i(a1)-P_i(a0)`, emitted only at the inverse
   position of `P_i(a1)-1`. Empty intersections emit no credit. Total across CTAs is512.
4. All this helper's inter stores for that physical job precede its final occurrence.
   Existing all-lane `__threadfence`, `__syncwarp`, then lane0 atomic credit is preserved.
   One CTA cannot credit another CTA's stores. No intermediate inter data is read before
   existing acquire polling observes the512 total. Same physical row/column arithmetic,
   BF16 rounding points and FP32 epilogue arithmetic remain unchanged.
5. No phase A compute depends on another CTA. Every positive A contribution publishes
   before its helper can wait on any B partial; no pending credit survives A. Thus after
   all finite A units complete, all512 counts become visible and B waits can complete.
   Common cooperative admission guarantees the complete grid can reside. Existing
   watchdog/error behavior is retained, not used as a correctness argument.
6. Existing kernel-wide joins/generation/reset/final read remain unchanged. No new
   counter or persistent buffer is introduced; graph replay lifetimes remain the common
   control's. Private harness states must preserve independent scratch and capture.

CPU2tests pass: allJ1..32,units256/512,grids1/7/132/136 coverage, clipped prefix/count/last
occurrence; source exact pin and no AST changes outside `_SKINNY_BLOCK`; unchanged counts
of all mbarrier/named-barrier/proxy/CTA-sync/rounding/MMA primitives. CPU fixtures prove
mapping/publication counts, not actual CUDA ordering or math qualification.

## Actual local sm90a compiler screen

Generated raw source SHA `a425ebdaf285456faaf1f641f074fd40622e8aaeb7aaa46ad652fce5216223eb`,
301803B. `tools/moe_decoder_compile.py` compiled both original and treatment with isolated
NVRTC/PTXAS12.8, no GPU execution. Host wrappers are removed by the documented extraction;
actual common cooperative launch/capture still needs H200 qualification.

|Actual skinny entry|Control|Band32|
|---|---:|---:|
|Registers|140|130|
|Stack /spill store /spill load bytes|0 /0 /0|0 /0 /0|
|Barrier count|16|16|
|Static full-entry instructions incl non-hot paths/NOP|3000|3408|
|HMMA BF16 /HMUL2 BF16|32 /128|32 /128|

No changed shared layout. Lower register count does not establish faster runtime or
achieved occupancy. Added variable-divisor address arithmetic, more activation loads,
more CTA-job credits, and possibly later A readiness are meaningful performance risks.
Static count is not a dynamic-hot-loop model. Unchanged main47/51 bodies compile at96regs;
main47 has existing12/112 spill bytes, main51 zero. No main treatment selection is allowed.
Compiler manifests/logs and opcode census are archived under
`evidence/2026-11-01-owned-band32/`; raw CUBIN/SASS remains in persistent WSL
`/root/.cache/okc-owned-band32{,-control}`.

## Gate and stop

Independent reviewer must clear exact mapper, metadata lifetime, clipped credit and
all-A-before-B proof before any check. Root owns any quota/action. The prospective private
four-arm harness uses held-out993701/993703/993709 disjoint from the census; identical
logical weights with independent repacks/scratch/inputs/captures, two binaries and two
private placements per binary. Common BF16 gamma and cooperative admission in both arms.
FP32 finite global.008/row.075 correctness, fresh/skew/all-remote/exactzero/restoration,
resource/capture/admission failures fatal. Score every load and all four control/treatment
crosses; require >=1% whole-six-load improvement in every seed and >2x duplicate null,
plus prospective per-case regression gate. No input timing selects the implementation.
No posthoc band-size sweep or selective seed policy. Failures close this exact hypothesis.
This checkpoint does not authorize dispatch, promotion, or public submission.
