# Owned projection sparse E7-XOR prototype

2026-11-01. Root requested this new owned track after `ff05d88`; reviewer
base `5d245b1`. No H200 job, paid resource or public submission by this worker.
**A distinct reversible prototype now exists; no performance win is claimed.**

## Corrected inventory and opportunity

`docs/context.md`'s old statement that 12-bit compression is not built is
obsolete. Pinned C2 already compresses every selected short BF16 route:
SK10, RD3 and CUDA-core GEMV3. The remaining short four router cells use
FP32 GEMV or the BF16 hi/lo representation; they are not BF16 compression
targets. Full30-cell latency/SGLang/score evidence remains the exact C2
`9eba96894ee1402e` map in the preceding WGMMA audit, with score1.5278064829557674.

| Prior mechanism | Owning evidence | What is already closed |
|---|---|---|
| Raw BF16 instead of12-bit SK |3843af6418a84a72|Raw loses0.6-3.7us; current compressed consumer already matters.|
| SK pair7/denser timing factorial |077b210c2087470d|Only+0.85% hypothetical score with a losing qkv32 arm; not an actual packed-stream result.|
| Actual pair7 scalar/vector/TOP7 GEMV |962d2f9c1d9644f7,4aef50bcd265448b,ee39ec933a8542d8|Exact bits/probes pass, but1-3.4% fewer physical bytes lose to decode cost; no new LUT/layout sweep.|
| Integrated original GEMV escape products |bac995b3b94541c4|Real residency gain, but serializing original extra FP32 products on a compute warp loses.|
| Register-direct RD16 and private cp.async ring |1759290da58047ab,bcabe48caa884e67,b103af15ba564ef6|Different consumer schedule loses; ring removes spills but not latency.|
| Compressed WGMMA reconstruction |4922f366a0c142ca|Raw twin already loses to Lt; software reconstruction cannot rescue it.|

The new target preserves SK's existing geometry/TMA stream. Current SK
escape warps wait for full slots, perform data-dependent table/X loads and
FP32 delta products, then release the empty barrier. The producer's empty
count includes these extra readers. Every CTA joins their cbar and folds
CORR_FLOATS into output. This dependency exists even when a particular
unit has no escapes. This is observable source/instruction evidence, not
an isolated measured escape duration or an HBM counter measurement.

The prototype repairs the escaped exponent directly in decoded BF16 B
registers before mma.sync. It removes separate delta products/correction
warps and their join, retaining SK's current TMA/consumer schedule. RD
already proves the idea of exact XOR repair in a different register/global
kernel; this is a new representation/placement in the incumbent SK pipeline,
not adoption of external standalone code or repetition of integrated GEMV
FP32 products. It changes FP32 summation order on escaped weights, so full
FP64/output-BF16 correctness gates remain mandatory.

The full30 score sensitivity is substantial enough to investigate: uniform
whole-operator cuts of3/5/10% across SK10 would yield1.011089/1.018707/1.038598
score factors. These are exact counterfactual score calculations from C2,
not escape-cost bounds, speedup forecasts or ranking promises.

## Exact source and API

Control raw bytes216576, SHA256
`5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4`;
normalized LF212002 bytes, SHA256
`7f4fd58cdd940f36c8d4d355f3de85687bc7f6ab04981842ce936db76d78adc2`.
Root's tracked `proj_sk_xor_control.restore(root)` reconstructs these raw
bytes from owning blob3044ee7 plus the authorized repeat comment. New
`tools/proj_sparse_xor.py::generate(raw_bytes_or_path)` accepts that result;
no generated payload reads source files or git at runtime.

Frozen prototype is310434 bytes, SHA256
`d23aa17be4cf9e7c5869803b3b4603d8e72395953687adc0287fb1c75738d586`.
Actual added CUDA string SHA256 is
`8cdbb4a99f226beb3f96ca662ae9d260bb7e002ef697337d8f80c075ccaa5068`.
Both added CUDA/CPP strings are readable raw multiline literals with exact
value assertions. Original C2's entire normalized prefix/AST and normal
prepare/kernel entry remain unchanged: this is OFF by default.

The generator exposes `xor_build(tag)`, `xor_pack(name,weight,variant)`,
`xor_runner(module,name,weight,variant,nearby=False,packed=None)` returning
`(callable,info)`, and `xor_expected_layout(name,variant,pmax)`. Variant0 is
the original control,1 the prototype. Each module contains both versions;
nearby configs clear only the original FULL-addressing bit. Runners own
private diagnostics/output; prepared immutable streams/tables may be shared.
Info includes exact cfg, private/immutable ownership, expected layout and
queried7-field resource vector. No source-global config mutation occurs
inside runtime build. Only an isolated offline generator module is changed.

## Representation and ordering proof

The original768-byte/512-weight low-byte+nibble unit stays unchanged. Each
stage appends a16-byte-aligned array of one u32 header per unit, in existing
`[warp,uu,nt]` order. Header low10 bits hold count0..512; upper22 bits are
the start into an immutable u16 record array. A record stores lane5, register3,
pair-half1 and exponent-XOR7 bits. XOR targets only bits8..14 of one BF16
half; its sign, low byte, neighboring half and NaN payload bits stay intact.
Records are unique per unit/lane/register/half. Header bounds and offsets
are checked; matrices outside the finite/sparse offer return None. The
internal transport core supports all bit patterns independently of that
offer. Original qkv row balancing and output rowmap are preserved.

The global ptab now stores packed u16 records, padding to an int32 boundary,
then the int32 rowmap. Only the new flag reinterprets pmax as that rowmap's
word offset. Every old shared table/correction use is conditionalized away
for SX; control uses the original ABI. New headers are included in the
existing bulk transfer and full-barrier expected bytes. There is one transfer,
not a new asynchronous side channel.

Header reads and all global repairs now precede load_a and the original
early empty arrive. The repair loop is warp-uniform; lane masks apply XOR to
statically named registers. No dynamic register-array index or new divergent
MMA participation is introduced. Putting repair before load_a avoids keeping
A fragments live across rare global reads. All main-reader proxy fences,
warp syncs, early/late empty arrives and final shared reduction barriers
remain. Only nonlaunched escape readers leave EMPTY_COUNT; their cbar wait
and CORR folding disappear because no such producer/storage is used. Rowmap
and patch records are immutable global inputs, with no reuse lifetime fence.

This format adds header traffic (4 bytes per512 weights) and global patch
reads. It does not promise lower HBM traffic. Stage alignment also costs
shared memory: most geometries save correction storage overall, while
slice8 grows about1-2KB depending on the old padded table. Actual admission
must decide, not a blanket assumption that added metadata is free.

## Executed validation and actual local code

Four owned CPU tests pass in2.356s: tracked raw/LF pins and unchanged normal
AST, all65536 BF16 patterns across121 windows, record domains, original
control shared-memory formula equality and actual source read/release order.
Seven independent MoE-worker tests rerun on final d23 pass in14.171s:
all10 actual packers/control rowmaps, every decoded matrix element through
an independent SK lane inverse, dense all65536-bit transport, max512-record
units, empty metadata and nonfinite offer rejection. Independent review
commits98696e0/8becd51 retain the proof/tests. Worker e1c558b additionally
reports a local RTX5070 bit-only execution of the exact repair helper through
eager and changed-input/header/record graph replays. That does not test SK,
TMA, Hopper MMA or H200 timing.

Local NVRTC/PTXAS12.8 sm90a compiles all30 unique selected/nearby entries:
zero stack/spill stores/spill loads. The rebuilt10 selected controls match
the prior pinned C2 local opcode/operand lists exactly after preserving
branch targets and normalizing only names. All SX HMMA counts match control.
Selected per-thread registers control->SX:

| qkv8 | qkv16 | qkv32 | O16/unpool16 | O32/unpool32 | pool16 | pool32 | slice8 |
|---|---|---|---|---|---|---|---|
|76->76|66->72|112->101|56->61|84->79|60->62|86->79|58->52|

Launch threads drop from288/256/416 to160/160/288 by removing escape roles.
Evidence records unrounded register-thread products, not claimed hardware
register allocation. Static code grows (qkv8:1160->2400, pool16:992->1824
instructions); the rare patch loop has a real instruction/data-dependency
cost. Per-thread register increases are disclosed. Actual CUDA occupancy,
local-memory and exact static/dynamic resource checks must precede timing.
The first after-A placement had larger register lifetimes and was replaced
with the source-backed before-A order; no register-cap or geometry sweep.
An initial undefined header-load helper was a corrected local compile error,
not a measured kernel failure.

`evidence/2026-11-01-projection-sparse-xor/` retains source/compiler manifests,
logs, selected resources/code census and PTX/SASS as evidence. Compressed
evidence files are not loaded by an entrant. The compiler excludes existing
host wrappers via SK_DEVICE_ONLY and uses disclosed fixed-width header
adapters; full torch-extension host linking and the new resource getter
have not run on H200.

## Next gates and stopping criteria

Root assigned the separate full30 paired harness to the attention owner.
Required: one unchanged C2 prepare/anchor, two independent builds with four
private runner/input/graph/diagnostic owners, all30 cells/three preregistered
heldouts, all SK10 plus nearby and exact one-hot probes, FP64/output-BF16 and
fresh replay gates. All10 targets remain planned, with no post-hoc dropping.
Expected static/dynamic/shared/thread/arrival extents must match derivation;
zero local/spill and actual resident blocks>=control are prerequisites.
Shared-memory deltas are explicitly allowed when expected and admission
does not degrade. Full30 score must exceed1% and twice independent null
pooled and on every seed, with the registered per-cell regression gate.
The owner's final registration governs details before dispatch.

No actual H200 latency, aggregate candidate score, counters or prepare
duration exists. Two additional builds plus packing/oracles may exceed295s;
immutable pack reuse reduces redundant work. Only six weight layouts are
distinct among ten shapes because paired16/32 routes share packed weight
geometry. Root is assessing any separately authorized one-arm measurement;
this unchanged-normal-entry prototype itself is not that candidate. Do not
attribute an unchanged C2 score to XOR, publish it as a winner, or bypass
resource/correctness/coverage gates. A valid loss closes this exact sparse
repair schedule; no automatic denser-code/cap/tile sweep.
