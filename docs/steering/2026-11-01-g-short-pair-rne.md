# Lane3 paired BF16 RNE: closed after the selected dependency screen

2026-11-01. One bounded new representation, source/CPU/local CUDA12.8 only.
No GPU execution, GPU-driver or official/service API call, timing harness, score, geometry/
cap/cache sweep, gamma change, fence rollback or production edit. Root owns
index and integration. This is **not** the closed gamma precompute prototype.

**No independent candidate emerges.** Pairing the required gate/up/down rounds
is value-preserving under CUDA's paired-RNE contract, but the actual selected
code adds permutation work and a dependency before SiLU. Tiny's non-NOP static
instruction count rises3; Tiny2 rises2. Resources are unchanged. Close this
exact intrinsic-based representation before any runtime harness; do not count
it as an active optimization or try another unpack/packing variant in this scope.

## Source ladder and exact hypothesis

Control607e `muse_decode_layer/cand/g1b4_stage_release.py`,869076 bytes,
SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`,
owns607eb52bdaf24160. G0 upstream raw1ec189dc…/job3a5f879df0d44825 lacks
the added gate/up and scaled-down BF16 stage rounds. Removing those required
rounds is not a safe restoration. This test changes their representation only.

`tools/proj_g_short_pair_rne.py::build(source,enabled=False)` returns exact607e
when disabled. Enabled scratch `.build/today-proj/short-pair-rne/on.py`:
870233 bytes, SHA256
`19cfb4eeeb0dde21b142a23a880e8f921dfd02a63806a05649c40e6cc54e6a16`.
It edits only pinned helper0 `b8ed8543…`, adds a fixed Tiny/Tiny2 compile flag
and rejects other active geometries at preprocessing. No host parameter format,
module selector by timing or shared-memory ownership changes.

The three source sites are one WS-helper gate/up pair, its down-output pair
and the compute-warp WSBL-tail down-output pair. h-loop unrolling yields six
pair sites per selected entry, replacing12 scalar RNE conversions. Scale
multiplication **remains before** each round; SiLU, its final BF16 store, gamma
formation/multiplication, stats, reduction and output rounding stay in place.
The source retains original expressions in the flag0 branches and proves exact
helper-source reversal. All other Python AST nodes are unchanged except the
two explicit compile-definition appends. No expert/finalizer flag is restored.

## Numerical representation proof and its limit

The new helper is:

```cpp
return __bfloat1622float2(__floats2bfloat162_rn(a, b));
```

Pinned CUDA runtime12.8.90 headers `cuda_bf16.h:327-347` specify that low16
bits correspond to scalar-RNE(a), high16 to scalar-RNE(b). The actual
`cuda_bf16.hpp:569-578` emits `cvt.rn.bf16x2.f32 ..., b, a` on SM80+;
632-638 widens those two corresponding BF16 halves. This preserves signed
zero/subnormal/finite/overflow/nonfinite conversion semantics under that API
contract, not a new approximate arithmetic mode. Header hashes are archived.

`test_proj_g_short_pair_rne.py`: **2 CPU tests pass in1.658s**. Independent
integer rounding models cover524288 paired binary32 representatives: every
high16 bit pattern times eight low-half representatives below/at/above the
halfway point, zero/extremes and NaN classification. Pair-half order and exact
scalar-round/widen bit results match, including canonical NaN handling in the
header's scalar conversion model. This covers rounding classes; it is not a
claim of executing all2^32 values or actual CUDA math on a GPU. Full native
kernel correctness and legal whole-layer common-control qualification are
unmeasured. Exact products and source-scope/fence/gamma preservation are tested
separately from that numerical model.

## Actual selected compiler result and dependency budget

Four fixed builds: Tiny/Tiny2 control and paired variant, actual PDL-false
`okc::moe_fused<false>` entry each. Existing NVRTC/PTXAS12.8.93 sm90a framework
and actual common/per-key defines are reused; no new toolchain. Device extraction
keeps the original prefix before Torch host wrappers, existing virtual stdint/
type-trait headers and declaration-only wgs_cta_of adapter. Host launch/API,
CUDA capture, driver and H200 execution are excluded. Disassembly uses existing
nvdisasm12.8.90; source-line output identifies the following chains.

|Selected entry|Control|Paired|
|---|---:|---:|
|Tiny registers /barriers /stack /spill store /spill load|147 /3 /0 /0 /0|same|
|Tiny static instructions, including NOPs|5272|5280|
|Tiny non-NOP instructions|5230|5233|
|Tiny2 registers /barriers /stack /spill store /spill load|147 /3 /0 /0 /0|same|
|Tiny2 static instructions, including NOPs|5328|5328|
|Tiny2 non-NOP instructions|5279|5281|

Launch bounds remain288 threads/min1 CTA. Actual CUDA admission was not queried.
PTX changes12 `cvt.rn.bf16.f32` into6 `cvt.rn.bf16x2.f32`; scalar widening,
other rounds, all explicit synchronization and MMA opcode counts are retained.
SASS changes12 F2F conversions into6 PACK_AB conversions, **adds8 PRMT** and
changes integer/address allocation. This is not an executable-instruction win.
No whole-SASS operand-equivalence claim is made for the changed representation.

The real gate critical path gets worse structurally:

```text
control: scaled_gate FMUL -> F2F -> IMAD(widen) -> sigmoid log2 multiply -> EX2
pair:    scaled_gate AND scaled_up FMUL -> PACK_AB -> PRMT -> IMAD(widen) -> sigmoid log2 multiply -> EX2
```

Control SiLU starts before the separate up conversion; its up multiply/round
overlaps the sigmoid sequence. Pairing waits for both operands and inserts a
PRMT before the gate's log2 multiply/EX2. Actual Tiny control PCs11850-118b0
and treatment11850-118d0 show this change. Removing six static conversion
instructions is therefore **not** removing six exposed critical-path steps.
The down-output pairs also need extraction; unchanged register count does
not make that work free. No sampled stall/counter/region measurement establishes
a cost for these chains.

## Full-six budget: assumptions exposed, not a forecast

Current weighted short-load latencies from owning607e are78.549045 and
115.990823us. Holding the other four loads and baselines fixed, saving3us
in both contexts of both short loads would give conditional factor1.010916763;
the equal cut needed for1% is2.753184us. These are arithmetic targets, not
measured or predicted savings.

Source `RB_A=512`, `RB_B=256` gives512A units and256B units per local skinny
expert job. T1 has at most8 jobs; T8 at most32 (each expert gets at most8
tokens). For an **explicit assumed132-CTA balanced-grid scenario**, per-CTA
assigned-unit ceilings and the intended pre-unpack conversion saving are:

|Load scenario|A/B unit ceilings|Intended saved warp conversion instructions|Exposed time needed per intended saved instruction to remove3us|
|---|---|---:|---:|
|T1, maximum8jobs|32 /16|64|46.875ns|
|T8, maximum32jobs|125 /63|376|7.979ns|

T1's gate executes only one h half for its valid token; down arithmetic can
cover both halves. T8 uses up to two. WSBL tails run on compute warps, and
helper/compute phases overlap, so summing their instructions as one serial
exposed chain is optimistic. Exact owning runtime grid, job counts and stalls
are not retained here; fewer jobs lower those instruction counts. The scenario
also grants an instruction saving that the actual PRMT/unpack code **does not
deliver**. It cannot justify ~3us removal in both short loads. No assumed GPU
clock is used to turn those counts into a latency claim.

`proj_g_short_pair_budget.py` and `budget.json` reproduce the arithmetic and
mark actual exposed time/latency forecast null. Whole-kernel static percentage,
cross-receipt expert-row gaps and this instruction scenario are not substitutes
for a component measurement. There is no credible independent >=1% candidate
from this prototype.

## Handoff and stop

Status **CLOSED_DEPENDENCY_BUDGET** for exact19cfb4ee, not a GPU performance
loss. Preserve CPU/source/compiler evidence under
`evidence/2026-11-01-g-short-pair-rne/` (compressed compiler evidence only,
not source/runtime packaging). No harness, GPU run, alternative unpack,
gamma mask, adapter or conversion variant is proposed.

The next changed G group to prioritize is the existing short publication/
final-counter lane, whose selected compiler screen actually isolates ordering
overhead; it is already owned separately and is not a new lane3 candidate.
Do not switch to unrelated standalone/dense tuning. Reopen a precision/dataflow
representation only for genuinely new source/codegen evidence that removes a
real dependency with a plausible whole-six budget while retaining every required
BF16 value and all progress/ordering contracts. Current common-control holds
remain in force.
