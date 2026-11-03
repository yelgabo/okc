# D32-ATTN-01 implementation evidence

Control ba9 and treatment 3de8f5ca. See the linked manifests for full hashes.

## Changed mechanism


The actual selected full-decode path is `attn_kernel<1,true,0>`, with 128 producer
and 128 consumer threads and three 64-key raw K/V stages. The control finishes
QK, online softmax and PV for a tile before starting the following tile's QK.
Its producer ring already overlaps data copies with that work.

The candidate computes QK for tile zero in a prologue. In each steady iteration
it acquires and patches the following tile once, interleaves its K-load/QK chain
with current softmax, performs current PV, then carries the following scores
across the loop edge. Every online max, sum, output-rescale and PV update remains
in chronological tile order. The final tile issues no lookahead acquisition.

This is a within-request change. The existing continuity candidate changes
cross-request scheduling and merge storage. Prior L8 XPIPE losses changed
WGMMA schedules; the L1 QK2 experiment changed accumulator chains within one
tile. The reviewed sources establish different implementations, not proof that
this candidate will win. Relevant history is in
[continuity](../../../2026-11-02-worker-attention-continuity.md),
[its reconciliation](../../../2026-11-02-worker-attention-continuity-reconciliation.md),
and `docs/runs-attention.md` around the XPIPE and L1 QK2 entries.

Compiled scheduling is narrower than the source proposal. In the saved treatment
[SASS](treatment-compile/device.sass), next-K `LDS.128` at
`0x6800` sits between current `MUFU.EX2` at `0x67d0` and `0x6810`. The next-QK
HMMA group at `0x6900..0x6cd0` follows the probability exponentials and precedes
current-PV HMMA at `0x70c0..0x74f0`. The compiler grouped QK math separately.
This establishes next-K load/exponential interleaving and carried next-tile
scores, not sustained simultaneous QK/softmax execution or saved cycles.

Potential benefit is a shorter exposed load dependency between tiles. Costs
include earlier full-stage waits, a longer live-score interval and prologue/
drain work. The bounded full-layer test decides their net effect. The 20-us
attention target remains a target. No latency estimate follows from this compile.

## Dispatch, preparation and numerical boundaries

The treatment is selected for batch 17 through 32, q_len 1, full context, page 1,
16 query heads, two KV heads, D64, zero RoPE theta, unit metadata cache scale,
and finite positive score scales. It supports the original valid length range,
including empty segments, one tile and partial final tiles. Existing cache-size
and integer-range checks remain outside this selector. Other metadata uses the
control. Preparation enables treatment only for compute capability 9.0 and
132 SMs; other devices keep control.

`_prepare_d32_attention` builds both control and treatment in the existing
preparation builder slot. The new module belongs to prepared state under
`d32_attention_module`. Timed selection cannot build it. Missing preparation
fails, and explicit use of that state's treatment module rejects unsupported
metadata, changed split/copy/cluster geometry, changed w0 or enabled PDL.
The selected tuple remains 264 flat CTAs, one workspace copy, no cluster,
noncooperative decode, w0 1024 and PDL off.

All other standard cells retain the original attention module and complete
launch argument tuple. All MoE, projections and input preparation stay control.
The original raw-cache writes, new-row patch, every-key masks, FP8 cache,
query representation, FP16 score/PV arithmetic and BF16 output boundary remain.
The half merge, finalizer, gate/norm and returned-state computation are unchanged.
The original 56-replay startup check is an expert check; it does not qualify the
new attention module. No new attention replay claim is made from that banner.

## Buffer lifetime and progress argument

| Storage | Writer and final reader | Publication and release |
|---|---|---|
| Current raw K/V slot | Existing producers/patch writers; current QK and V/PV readers | Existing full wait and patch join before reads; existing warp join and all-consumer empty arrival after current PV |
| Following raw K/V slot | Existing producers/patch writers; lookahead QK and that tile's later V/PV | Acquired once before lookahead QK; remains held until that tile's own PV retirement |
| Carried scores | Each consumer lane's QK; same lane's later softmax | Registers, no cross-thread publication; read only when the guarded lookahead wrote them; reset per segment |
| Query and new-row staging | Existing consumer preparation; unchanged QK/patch readers | Original staging waits and joins retained |
| Page-index ring and barriers | Existing producer TMA/copies and page readers | Producer loop, phase arithmetic, final page-reader join and per-segment initialization unchanged |
| Half-merge and global workspace | Existing consumer merge/finalizer | Original joins, 18-float merge and contribution counts unchanged; producers drain at the original segment boundary |

The lookahead condition and patch condition are uniform across the consumer
group. Acquisition and its patch join sit outside the per-query `active` branch.
MMA calls remain warp-uniform. No new shared allocation or proxy access exists;
no existing fence, full/empty handshake, WGMMA wait or segment join was removed.

With three slots and one-tile lookahead, publishing tile i+1 can depend only on
retirement of tile i-2. A lagging consumer can finish that older tile using its
already-published successor. Current/next slot ownership therefore creates no
source-level producer/consumer wait cycle. This reasoning retains all-consumer
empty arrivals and the unchanged producer order. Sampled CPU events check the
model, including early-release and one-slot deadlock negatives. They are not
mutated-kernel runs or an exhaustive CUDA scheduler proof.

## Observed checks and limits

Six focused tests pass in 7.612 seconds in
[cpu-tests.txt](cpu-tests.txt). They execute actual extracted
final bindings and preparation with CPU stand-ins, compare all 12 standard
launch tuples, check 17/24/31/32/33 neighbors, scale/shape fallbacks, missing
modules, module isolation and launch-drift negatives. They also verify source
derivation, exact inverse CUDA boundaries and sampled progress events.
Fresh corresponding control/treatment builds use cached CUDA 12.8.1 NVCC/PTXAS
for sm_90a and instantiate exactly `attn_kernel<1,true,0>`. Original selected
defines and Torch half/BF16 defines are retained. Both report 128 registers,
three barriers, zero stack and zero spill stores/loads. Both constant layout
witnesses are `[61440,256,3,56,200,2]`, for shared-struct bytes, threads, stages,
producer/consumer register allocations and minimum blocks. Dynamic shared
allocation remains `61440 + 1024 = 62464` bytes. These are compiler/layout
observations, not actual occupancy or admission measurements.

The adapters strip Torch host includes, add CUDA runtime declarations and the
selected explicit instantiation, and append a constant layout witness. They
leave selected device bodies intact. This is device compilation; the full Torch
host extension, preparation budget and H200 runtime remain unchecked. Complete
commands, image identity, adapted sources, PTX/CUBIN/SASS, logs and hashes are in
[compile-summary.json](compile-summary.json) and its two arm
manifests. Static PTX barrier sites differ by one because lookahead has distinct
prologue/steady acquisition sites. A site count is not a dynamic execution count.

The first receipt reader rejected a PTX initializer that omitted trailing zero
bytes after successful control compilation. Its
[failure record](compiler-parser-failure.json) is preserved.
The helper now pads omitted initializers and tests overflow rejection. Both arms
then completed the corrected compile screen. This was a receipt-parser failure.

An independent read-only review checked the actual CUDA change, numerical
ordering, patch-join uniformity, slot retirement, progress and host guards. It
found no required source fix. Compiler outcomes and this report receive a final
readback before publication. GPU math, causal/cache outputs, fresh replay,
guard buffers, runtime resources and complete-layer timings remain NOT_RUN.
