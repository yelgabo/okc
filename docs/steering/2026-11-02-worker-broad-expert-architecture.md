# Broad expert-engine decision: no adequate demonstrated path

November2,2026. P1 allocation **588847c9**, complete control
**ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab**,
fresh layer receipt **cc474883117947c0**, public incumbent **607eb52bdaf24160**.
One bounded source-only decision, delivered before its45-active-minute bound.
No kernel, harness, compiler, installation, child worker, GPU/Driver, challenge
CLI/POST, default or release change. Existing6e181attention remains stopped.

## Decision

**NO ADEQUATE PATH FOUND for a new several-percent net layer allocation from
these two complete dataflows.** Do not dispatch a Humming wrapper retry or build
an expanded-weight engine on this evidence. This closes these proposed bets at
the evidence gate, not the expert-engine design space or the top100 objective.

Two materially different designs were compared:

1. Replace the fused native expert engine with current Humming grouped-contiguous
   W4A16, retaining BF16 activations and all Muse nonlinear/normalization steps.
   The current engine is genuinely different from the previously measured
   Humming version, but no same-format/shaped cold result establishes the large
   GEMM gain needed after replacing fusion with multiple callable boundaries.
2. Decode immutable FP4/block scales once to the exact BF16 operand words and
   replace timed conversion with a BF16 grouped engine. This removes timed weight
   decoding, but grows logical weight bytes3.56x versus ordinary packing and3.66x
   versus eligible SC6. It retains both dense products and all nonlinear work.
   Under broad cold routing the traffic cost contradicts the proposed saving
   budget, even before intermediate/normalization overhead.

**Measured saving: unknown, not0%.** Neither design was executed. We found no
credible latency estimate or exposed-time evidence that earns several-percent
net layer benefit. A10x improvement in the removed decoder alone would not make
the complete operator10x faster. No implementation allocation is requested.

## Complete immutable intake and what changed

Inspected Humming at **948e6fd547e703065be51529ac1db5569256774e**,
the previously identified primary source, through its current
[vllm-project repository](https://github.com/vllm-project/humming/tree/948e6fd547e703065be51529ac1db5569256774e).
The GitHub API tree is untruncated,309blob paths. Retrieved153complete source/API/
header/license/test files,1553169bytes; every source is checked against its Git
blob SHA and separately SHA256-pinned. Third-party bytes stay only in ignored
scratch `.build/worker-broad-expert-architecture/`, not in committed evidence;
they are not imported or executed. The complete inventory, HTTP provenance and source
hashes are in [source-manifest](evidence/2026-11-02-worker-broad-expert-architecture/source-manifest.json).
Detailed review focused on the call/dependency chain below; acquisition of all
headers does not mean every line of every helper received an independent proof.

Compared with the actual earlier installed **636ba85648c30ae2c2bfb335c9399593a67ecc1d**,
the pinned version is103commits ahead. Comparison metadata and five specific
old-path retrieval hashes are archived; patches/source bytes remain in scratch.
The old sm90_policies.py path404s;
that is an absent old file, not an unavailable current repository.

Important concrete changes include grouped persistent raster prefix construction
and expert lookup in scheduler.cuh, Stream-K alignment/slice guards, and newer
SM90 WNA16 compact/overlap policies. Current compact policy can remove dedicated
producer warps/TMA under explicitly constrained block-K64, tile/residency and
pipeline checks. It is not merely the old library with a renamed wrapper.
Those are source differences, not measured throughput gains. Generic persistence,
continuous rings and deferred bookkeeping already exist in our engine; they are
not counted as new candidates.

The first-party [architecture announcement](https://vllm-project.github.io/2026/10/19/humming.html)
motivates comparing complete pipelines. Its performance plots use other formats,
shapes and baselines; none is credited as a Muse result. Its fast Hopper
MXFP4-to-FP8 path is not this operator's NVFP4/BF16 combination.

## Actual control: all four longer loads, both contexts

The complete layer enters experts after BF16 pre-expert normalization and FP32
router/top8 selection. Global IDs are converted to local32-expert IDs; inactive
slots remain-1 and route weights stay FP32. `_fork_native_moe` is the final
state-aware dispatcher, not the earlier obsolete `_moe` definition.

| Tokens / load | Actual control route | Proposed A route | Proposed B route |
| --- | --- | --- | --- |
|16 /D16|MID skinny helper0df, NW8/TINY1/SPLIT1; its own scheduling, not WGS|Grouped-contiguous BF16/NVFP4 G1 and G2|Predecoded-BF16 grouped G1 and G2|
|32 /D32|B4 WGS main plus B4FIN|Same engine replacement, separate small-expert config|Same BF16 replacement, separate small-expert config|
|128 /V16|MAIN WGS, NW16, SC6 when eligible|Same grouped replacement with actual routing counts|Same BF16 replacement|
|256 /V32|MAIN WGS, NW16, SC6 when eligible|Same grouped replacement with actual routing counts|Same BF16 replacement|

Each row applies to sliding and full contexts. T1/T8 and noneligible fallback
remain unchanged; no MID performance inference is inherited from a WGS result.
Current module keys, host/CUDA flags and source closure were already independently
recorded in the exact [component adapter](2026-11-02-top100-moe-adapter.md).
This decision reuses that complete control evidence rather than inventing another
compiler/qualification exercise. The control native source is c7c201ae... and
skinny helper b8ed8543... in that manifest; their complete parent file is pinned
above. No compiled resources of an unimplemented replacement are asserted.

Control preparation retains original input banks plus packed skinny/unit/fat/WGS
copies, gate/up row interleave, K permutation, scale-byte PRMT and conditional
lossless SC6. `_fork_native_ma_prepare` packs32experts, retains FP32 global scales,
raw BF16 post-norm weight, lookup state and private guarded scratch. Every target
T fits the384-token scratch limit, so the Python chunk loop makes one native call.
The native callable owns routing/setup, both phases, SiLU, partials, expert-wide
normalization/combine and cleanup under the current cooperative/progress guards.

Required numerical sequence survives either proposed engine: BF16 gate and up
outputs after FP32 dot/global scaling; FP32 SiLU/product with the required BF16
intermediate; BF16 down-output boundary; effective layer BF16(1+raw norm weight),
per-expert gamma/normalization with1e-8 epsilon; FP32 route weighting/combine and
the final BF16 output. The expert-wide norm occurs before expert contributions
are summed. It cannot become one norm after mixing or disappear into a per-tile
GEMM epilogue. Old standalone Humming gamma corrections are not the layer ABI.

## Humming's full relevant path and applicability

Call chain reviewed: HummingLayer/schema load -> prepare_layer_config,
check/pad/transform/repack -> prepare/config dispatch -> humming_forward ->
ops.humming_gemm -> HummingKernel/launcher -> standard or warp-specialized
kernel -> scheduler -> operand G2S/S2R loaders -> mainloop dequant/WGMMA ->
epilogue/reduction/store. Exact inspected paths/hashes are in decision.json and
source-manifest.json; the complete third-party bytes stay in ignored scratch.
Primary anchors include [MMA type selection](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/humming/config/mma.py),
[scale schema](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/humming/schema/humming.py),
[grouped scheduler](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/humming/include/humming/scheduler.cuh)
and [SM90 policies](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/humming/tune/sm90_policies.py).

| Obligation | Source result and remaining boundary |
| --- | --- |
|Hopper/BF16|config/mma.py:get_mma_b_dtype returns BF16 for BF16A; SM90 defaults WGMMA. sm90.py admits16-bit operand types. No native HopperFP4 MMA is being assumed.|
|NVFP4 scales|schema/humming.py supports float4e2m1, E4M3 group16, tensor secondary FP32 scale; epilogue_arith applies tensor scale before output narrowing. Full accumulation/exponent-offset/rounding equivalence still needs target validation.|
|MXFP4 fast path|dequant_fused.cuh explicitly targets E8M0/MXFP4 conversion to FP8/INT8. It is not an E4M3group16 BF16 shortcut. Quantizing activations to FP8 would change this problem's boundary.|
|Expert layout|Scheduler has INDEXED, GROUPED_CONTIGUOUS and GROUPED_MASKED paths,32-expert counts/offset prefix compatible in principle; local IDs must be mapped by current shard, never treated as a full256bank.|
|Zeros/empty|No positive-scale assumption is justified. Zero-count experts generate no grouped tiles; inactive output slots must remain masked/defined, especially after occupied->empty replay. Zero weights/scales and special values still need complete numerical checks.|
|Tile fit|WNA16 policies use block-K64 and output-N128/256 alternatives with16/24/32/etc token tiles, constrained by resources; no single actual target tile or resource set was instantiated here. Schema support alone does not choose a fast tile.|
|Muse norm/nonlinearity|Generic GEMM output processing handles scales/bias, not the complete Muse post-expert norm. The complete caller must retain SiLU, both BF16 interfaces and the norm/combine work externally or implement a separately proven fused operator.|

Shape metadata is sufficient for preparation; routing counts remain runtime data.
Packed weights/scales may be transformed once; runtime activation/route/output
caches are not proposed. The general input-independent prepare/repacking rule
does not prove an arbitrary full-function cache legal. No new outside-work
exception is needed for inspecting these engine designs; release eligibility
and notices would still need checking before adoption. Humming's retrieved
license is retained; no donor code entered a release.

## Design A: replace the whole engine with grouped-contiguous W4A16

Before: one current persistent/native expert callable and its own scratch/state.
After: count/prefix/rank local routes -> gather valid BF16 rows into compact
expert segments -> Humming G1 W4A16 BF16 output -> exact gate/up SiLU and BF16
intermediate -> Humming G2 W4A16 BF16 output -> expert-wide norm/weighted scatter
and BF16 output. Keep router, pre-expert norm and other layer work fixed.

Full replacement costs optimistically start at five launches if routing/gather
can be fused: gather/metadata,G1,SiLU,G2,norm/scatter. A straightforward separate
count/prefix/placement implementation raises that count. This optimistic fusion
is a proposed cost allowance, not a supplied implementation. Grouped-contiguous still has MMA token-tile
padding and Stream-K reductions; it does not turn all sparse routes into dense
useful rows. Must allocate runtime metadata, padded grouped X/G1/H/Y, reduction
locks, output and norm workspaces, plus separately packed weight banks and JIT
modules. Zeroed tails, route ordering, capture ownership and every empty replay
remain obligations. No numerical boundary or ordinary-kernel ordering guarantee
may be deleted to pay those costs.

For R valid local assignments, minimum read+write traffic for X/G1/H/Y is
**81920R bytes**, before output/norm rereads, padding, partials and metadata.
At the illustrative R=T this is1.31/2.62/10.49/20.97MB for16/32/128/256.
Fully provisioned masked capacity32*T consumes20.97/41.94/167.77/335.54MB for
those four BF16 arrays; compact allocation shrinks capacity/traffic but cannot
erase the matrix arithmetic or every padded-row operation. Runtime Humming
register/shared limits have not been measured and are not borrowed from ours.

This could conceivably remove exposed decode/scheduling inefficiency in a new
complete engine. It does **not** remove10x mandatory work: the same FP4 values and
E4M3 scales still become BF16 operands for the same two dense products. New
raster/compact policies may matter, but their source does not price the gain.

Prior valid **ae8993f4e3e1435c** default GROUPED_MASKED comparison passed math
but lost145–184us at128 and203–235us at256. Its G1+G2 activities alone380.511us
exceeded its matched owned217.152us complete endpoint. Glue-only fusion therefore
cannot repair that version. New948 differs from installed636; this is **not** a
blanket rejection of every newer grouped configuration. However, roughly halving
those old GEMM activities merely to catch that old control is a substantial
unexplained requirement, not evidence for several-percent net gain over current
ba9. Old exact configs were truncated from receipts; they cannot be reconstructed
or causally compared with today's policies from the retained profile alone.

## Design B: prepare exact BF16 operands, change the timed engine

Before: packed FP4/E4M3 stream, per-tile conversion/scale decode, BF16 TensorCore
products. After: preparation materializes **the existing decoder's BF16 operand
words**, including its exponent-offset convention, retaining FP32 global scaling
in the original post-dot location; timed grouped BF16 SS/register-load products
replace packed dequant mainloops. This is a weight-processing/base-engine change,
not a decoder helper spelling change. SiLU, gate/up/down BF16 boundaries, norm
and route combine remain as in A or a separately proven whole-output-owner kernel.

Storing mathematically dequantized weights with global scale folded in is **not**
accepted equivalence: it can change operand and accumulation rounding. Even an
exact stored operand does not prove that a different GEMM reduction/split order
has identical complete BF16 gate*up outputs. No FP8/FP16 downgrade is proposed.

All32experts contain1610612736logical elements across W13/W2. Ordinary FP4+one
scale/16 uses905969664bytes; eligible SC6 uses880803840bytes; BF16 expansion
requires3221225472bytes (3GiB), excluding retained original/packed banks.
The timed scale/nibble decoder can become arbitrarily cheaper because it moves
to preparation, but the weight stream grows3.56x/3.66x and the two GEMMs remain.
Useful dot work remains100663296FLOPs per valid local assignment. Preparation
must read originals, allocate/write3GiB and perhaps transpose it; peak memory and
preparation wall time are extra unknowns, not free costs. Additional shared BF16
staging, tile parallelism, partial reductions and norm traffic still require a
complete implementation/progress design. No stage/resource claim is assigned.

To test this traffic bet without pretending a routing sample was measured, use
uniform independent top8 across256global experts: expected local routes=T and
active local experts=32*(1-(31/32)^T). The resulting **conditional** one-pass
logical weight stream is:

|T|Expected active experts|SC6 logical MB|BF16 logical MB|Ideal BF16 HBM stream us at4.8TB/s|
|---|---:|---:|---:|---:|
|16|12.745|350.82|1282.98|267.29|
|32|20.414|561.90|2054.96|428.12|
|128|31.450|865.67|3165.88|659.56|
|256|31.991|880.54|3220.27|670.89|

[NVIDIA's H200 specification](https://www.nvidia.com/en-us/data-center/h200/)
gives4.8TB/s peak bandwidth. These are optimistic stream times **if those bytes
traverse HBM**, not measured bandwidth, physical-traffic counters or universal
runtime bounds. Actual routing, repeated tile loads, L2 and overlap differ.
At128/256 this conditional expansion cost already exceeds the entire fresh
261.7/351.1us weighted layer endpoints, before any compute/launch/norm. Thus the
cheap-decoder story supplies no credible broad-cold end-to-end win. A demonstrated
large reuse/cache regime or exact lower-byte representation would change this
choice; peak math speed or spare memory capacity does not.

## Net six-load requirement and later discriminator

Exact algebra uses C=(46*sliding+16*full)/62 and equal-six mean log ratios, with
baselines and all other work held fixed. In a proportional-to-current-expert-
activity sizing model, required **net weighted endpoint savings** are:

|Desired layer gain|D16 us|D32 us|V16 us|V32 us|Fraction of activity hypothetically removed|
|---|---:|---:|---:|---:|---:|
|3%|6.120|6.633|12.857|17.361|8.798%|
|Sampled cutoff6.5763%|12.849|13.926|26.996|36.453|18.472%|
|10%|18.776|20.350|39.448|53.267|26.993%|

These are requirements, **not exposed savings or predictions**. Added metadata,
launches, conversions, padding, memory, norm, finalizer and integration effects
must all be paid before these numbers are counted. Neither A nor B has that
evidence. A10x main-engine scenario would be material if physically achieved;
we found no corresponding removal of necessary BF16 computation/bytes here.

If captain later obtains genuinely matching current-Humming benchmark/config
evidence, **A is the stronger reserve than B**, because it preserves compact
weights and has real source changes. The smallest subsequent allocated target
experiment would compare the complete component callable with current ba9,
fresh matched weights/routes/private states and retained post-replay outputs,
both contexts/all four long loads plus short regression controls. Include every
adapter/SiLU/norm/scatter call in the timed endpoint; a G1/G2-only result cannot
qualify replacement. Independently prepared duplicate controls/nulls and frozen
heldout banks are required. First stop on semantic/resource/setup failure;
otherwise stop if net savings cannot reach the3% sizing row consistently after
null/regression costs, then separately test the complete layer before release.
No harness or target call is built or authorized by this proposal.

Strongest immediate next alternative is **a complete measured, same-format
engine/base with a wider operator scope**, not another isolated flag: require
source-linked BF16/NVFP4 cold data at comparable32expert/4096 dimensions and
few tokens/expert before allocating an integration. The retrieved pinned tree
contains no same-shape Muse result establishing this case. A targeted complete
comparator intake could change the base; repeating the unchanged repo census
cannot. This is an explicit remaining evidence gap, not a discovered winner or
request to send another speculative small check.

## Reused closures and handoff

Whole-G0's historic advantage is confounded by different seeds/device/config and
its weaker intermediate/order paths; it cannot be reinstated as an earned gain.
MonoMoE's complete FP8/fixed-shape design lacks this Muse norm and contract; its
generic persistent overlap already exists here. Prior helper/gamma/SiLU/RMW,
mailbox/staging/resource and shortT1/pre-gamma closures remain exact. No oneCTA/
expert proposal hides the32/132SM underfill behind a persistent label.

Reproduce [analysis.py](evidence/2026-11-02-worker-broad-expert-architecture/analyze.py)
for pins, byte/FLOP counts and score requirements; [analysis.json](evidence/2026-11-02-worker-broad-expert-architecture/analysis.json)
records all assumptions. [Intake](evidence/2026-11-02-worker-broad-expert-architecture/intake.py)
and version comparison only retrieve source; no foreign code is executed.
Captain owns independent review, next main-computation allocation and README.
This worker returns the finite P1 decision; no further implementation or quota
action is automatic. The objective remains unachieved.
