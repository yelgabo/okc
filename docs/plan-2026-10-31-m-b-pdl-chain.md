# M-B: PDL chaining of the layer's glue and projections at T <= 32 (2026-10-31)

Requested by the coordinator after F-LAYER-0. Idea learned from alygg77/okc-muse-spark-kernels
@ 6cae49a (`kernels/muse_decode_layer.py`, read in the scratchpad only); every line here and in
the tools is our own code. Donor code is never pasted or submitted (user rule, 2026-10-31).

## Evidence that reopens the family

`docs/ruled-out.md` closed "PDL across our own chain" on 179cecdac87b4872 / e8b74a3519a94bc6:
every launch PDL with weight prologues before the wait moved the step by 0-3 us, inside the
noise of that time. New evidence (measured, F-LAYER-0 four-arm checks 20abe6291eb04ed6 and
00607fc6f8c84c77, ABK tables): the donor's per-step kernel sum exceeds its span by 7-10 us at
d1s, v1s, d16s and d32s (sum 89.5 / span 82.2, 122.2 / 112.3, 151.4 / 141.4, 180.0 / 169.7),
while ours exceeds it by 0.2-0.5 us at d16s / d32s and not at all at v1s (153.2 / 152.7,
190.8 / 190.6, 123.8 / 127.1). At d1s our 15.6 us comes from the side streams, not from PDL.
So the donor's kernels do overlap where ours do not, by about the size of the gap at those
shapes. Two things differ from the closed attempt: the waits now sit after a real weight
prefetch (M1's sk path issues its first ring of weights before the wait, which the closed
attempt did only for some kernels), and the decision uses the stable-pick four-arm harness,
whose nulls (0.03-0.12%, e582fde557114062) can resolve a 0.5% change. Whether overlap is the
cause of the donor's lead, or only accompanies faster kernels, is a hypothesis.

## Boundary inventory at T <= 32 (this layer, layer_m1rt51s)

| boundary (primary -> secondary) | PDL today | what the secondary can load before its wait |
|---|---|---|
| previous step -> qkv | no (graph start) | nothing to overlap inside one replay |
| qkv -> attention | wait inserted, launch plain (`_ONE_PDL = False`) | K/V cache rows below `seq_lens`, page table (harness inputs) |
| attention -> o | M1 (o is an sk PDL secondary at 8-32 tokens; cgemv at 1) | o weights, escape table |
| o -> attention boundary | yes (boundary kernels are PDL secondaries) | norm weights |
| boundary -> front (skf) | M1 "o+front" | pooler + router weight pieces |
| front -> route -> MoE | yes | MoE weights prefetch (L2 hint), mbarrier init |
| MoE -> unpooler | **no** | unpooler weights (**slice 1**) |
| unpooler -> FFN boundary / norm | yes | gate and norm weights |
| 1 token: every projection is cgemv | no | weights could be loaded to registers before a wait |

## Slice 1 (built): the unpooler as a PDL secondary of the MoE

`tools/dl_mb_unp_patch.py` on the stable M1 file: a fourth M1 mode `o+front+unp` launches the
unpooler sk pick with bit 16 at 8-32 tokens. No device code changes: the SK_PDL build already
compiles every single-CTA sk config with the PDL preamble M1 uses for o.

Provenance (what the unpooler reads before `griddepcontrol.wait`, from the sk kernel source in
the file): the first min(NSTAGE, stages) weight stages (`bulk_g2s` from the packed unpooler,
written in prepare), the escape table (`ptab`, prepare), the tensor map (a kernel parameter,
`prefetch.tensormap`), and shared memory it initializes itself (mbarriers, the zero row). x, the
MoE's output, is copied by TMA only after every thread's wait (`issue_x` after the wait), and
each stage's full barrier expects the weight and x bytes together, so no stage completes on
the weights. The kernel issues no global store before the wait ("every thread waits"), so a
reused allocation cannot be written while the MoE still reads it. Nothing the MoE (or the W4
chain, when no fused variant is picked) produces is read before the wait. The donor's E2
pattern (a secondary reading the o projection's output before its wait while its primary is
the pooler) has no counterpart: no secondary here reads any in-step output before its wait.

PTX ordering: the same chain as docs/steering/2026-10-30-m1-ptx-ordering.md. The MoE writes x
with generic stores; `griddepcontrol.wait` returns after the primary grid completed and its
memory operations are visible to the dependent grid; thread 0 then executes
`fence.proxy.async.global` before its TMA copies of x (the reader-side proxy fence the M1
review asked for), and the producer lane does the same before its later copies. The same
instructions run for o today; only the primary differs.

Gate: M1's fresh-input gate now replays `M1_GATE_MODE = "o+front+unp"` (the superset of every
mode's PDL launches), with the census skip, bounds and negative controls unchanged; the M1
self-test also launches every sk unpooler pick with the bit and requires bitwise equality with
the plain launch. A failure turns M1 off for the run, as before.

Pick: the stable rule (9 rotated rounds of whole-step loads per mode). The measurement build
`layer_mb1s` prefers `o+front+unp` at 8 / 16 / 32, so a tie runs it and a loss beyond the noise
leaves it; SUMMARY `stable m1 T=...` prints every mode's paired saving against it.

Expected effect (hypothesis): small at 8 tokens, where the unpooler slice is already
L2-prefetched during attention; at 16 / 32 the unpooler streams the whole 8192 x 4096 matrix
(20-22 us) and the first ring's latency (about 1-2 us) can hide under the MoE's drain. That is
at most about 0.5% of d16 / d32 / v16-free loads, so slice 1 alone is not the donor's 7-10 us.

## Next slices (not built)

1. Attention as a PDL secondary of qkv (`_ONE_PDL = True` per bucket): its K/V loads already
   precede the inserted waits (audit table in docs/runs-decode-layer.md). Open question: the
   cooperative modes (1x1 full and 1x8 full plan mode 11) launch with the cooperative
   attribute, and the probe drops PDL when the driver refuses the pair; progress must not
   depend on co-residency with qkv's CTAs. Needs qkv to trigger dependents early.
2. Early triggers (`griddepcontrol.launch_dependents`) in the primaries that do not have them
   (the MoE fused kernel at the end of its main loop, the sk projections after their last
   weight stage): without a trigger a secondary starts only when every primary CTA exits.
3. The 1-token chain (cgemv qkv / o / pooler / unpooler, 35 us of the d1s step): a cgemv PDL
   build that loads its weight registers before the wait.

Each slice: provenance table, the M1 gate with the census skip, the PTX note, SASS counts
against the previous build, and a four-arm bench against release_s.

## Measurement

Four-arm bench c8d1e57261dd4901: inc = release_s, cand = layer_mb1s, all six loads, 8 rounds,
seeds 260001+, order seed 79, `--dup=own`. Read against 3aee22ba4a404c8a (the same inc against
layer_m1rt51s): the slice's effect is the difference of the two pooled ratios per load, which
is only meaningful where each bench's nulls are under the difference.

Result (c8d1e57261dd4901): the slice lost inside prepare. Both cand copies left the preferred
o+front+unp at 8, 16 and 32 tokens (`{8: 'o+front', 16: 'o', 32: 'o'}`), which the stable rule
does only when another mode is faster beyond max(0.3 us, 2 x MAD) in at least 8 of 9 rounds.
The timed bench then ran layer_m1rt51s's policy and read pooled 1.0106 against release_s, a
second reading of that candidate (3aee22ba4a404c8a read 1.0052). Slice 1 is closed in
`docs/ruled-out.md` with its revisit trigger.

Next, in order (no quota left this round for the layer):

1. Print the M1 tuner's per-mode stable notes in one compact line that the bench keeps in its
   tail, so the next slice's whole-step savings are visible even when it is not picked.
2. Slice 2: an early `griddepcontrol.launch_dependents` in the fused MoE kernels after their
   last weight stage and in the sk projections, with slice 1 retested on top. The
   co-residency question: a trigger lets secondaries occupy SMs while primary CTAs still run,
   so a grid-synchronized primary must trigger only after its last grid barrier.
3. Attention after qkv (`_ONE_PDL` per bucket), then the 1-token cgemv chain.

Diagram: [plan-2026-10-31-m-b-pdl-chain.html](plan-2026-10-31-m-b-pdl-chain.html).

## Slices 2 and 3 (built: `tools/dl_mb_chain_patch.py`, `layer_mb2s`)

Built on layer_m1rt51s. M1's mode becomes a set of switches joined by "+", and the stable M1
tuner times `M1_MODES` = off, o, o+front, o+front+trig, o+front+unp+trig,
o+front+unp+trig+attn per bucket against the incumbent's preferred mode; the gate replays the
last (every switch on). Switches:

| switch | what changes | where it is read |
|---|---|---|
| unp | the unpooler sk pick launches with the PDL bit (slice 1) | `_m1_pdl` in `_proj` |
| trig | every sk / skf launch at that bucket sets bit 17 of its config; `sk_fwd` / `skf_fwd` pass it as bit 29 of M; `sk_kernel` (built with SK_PDL, single-CTA) executes `griddepcontrol.launch_dependents` at entry when it is set | `_proj`, the two front call sites |
| attn | `_attn_one` passes PDL for the attention launch at 9-32 tokens; the host drops it on cooperative modes (`pdl && !coop`); the launch is counted by the M1 census | `_attn_pdl` |
| (always) | both fused MoE kernels (the four skinny copies and `moe_main_kernel`) execute `griddepcontrol.launch_dependents` right after their grid barrier's `__syncthreads()` | device |

Trigger placement (from a read-only review of the kernels, layer_m1rt51s line numbers): the
skinny MoE's last inter-CTA wait is the grid barrier ending at 10110, mode 51's at 13146; the
fused MoE kernels had no trigger before. `sk_kernel` has no inter-CTA wait in its single-CTA
configs (the FUSE last-block reduction counts with `atomicAdd` and never waits), so its entry
is safe. Both MoE kernels run one CTA per SM on a grid of all SMs (persistent by grid size, not
cooperative). Shared memory leaves room for an unpooler sk CTA beside a MoE CTA only at 16
tokens (skinny MoE about 155.5 KB; unp16 about 69-75 KB); at 8 and 32 tokens a secondary CTA
can only start on an SM whose MoE CTA has exited, so there the trigger hides launch latency
only (computed from the structs, not measured).

PTX: `griddepcontrol.launch_dependents` only lets the dependent grid be scheduled once every
CTA of this grid has issued it or exited; it orders no memory. `griddepcontrol.wait` waits
until the prerequisite grids have completed and their memory operations are performed and
visible to the current grid (PTX ISA, griddepcontrol). So a trigger can change what runs
concurrently and what a secondary may read before its wait, never what it reads after. Every
pre-wait read in the chain is listed in the next section.

Progress: a secondary CTA can only be placed after every primary CTA has triggered or exited,
so it never takes an SM a primary CTA still needs; the MoE triggers after its grid barrier,
so its barrier cannot be affected at all.

## Slice 4 (designed, not built): the 1-token chain

At one token the projections are cgemv (`_SKP_GEMV_SRC`): qkv `cgemv40<256,2>`, o
`cgemv<64,1>`, the FP32 front's pooler `cgemv<512,1>`, the unpooler slice `cgemv40<128,2>`;
none has a `griddepcontrol` instruction and `cgemv_launch` sets no launch attribute. Design:
a runtime flags argument (bit 0: PDL launch and wait, bit 1: trigger at entry); in
`cgemv_body` the compute warps issue their weight loads (low bytes and codes) and then wait,
and the escape warp loads its first table entry and then waits, before any read of x; x is read
with `ld.global.cg` instead of `__ldg` in the PDL build, because the non-coherent path is
specified for data that is read-only for the kernel's lifetime, which a PDL secondary's input
is not; the host launches through `cudaLaunchKernelEx` with the programmatic attribute when
bit 0 is set. No CTA waits on another, so the entry trigger is safe.

Two obstacles keep it out of this round. The M1 tuner's buckets are 8 / 16 / 32 and one token
falls in bucket 8 (the v1 shapes time it), so the 1-token chain needs its own bucket and its
own whole-step rounds on d1 shapes. And the M1 gate would then judge 1x1 sliding and full,
the shapes where it failed on a one-replay BF16-rounding difference with a bitwise control
(d6c30f88c617438b, 6ce5be0ba29642ea), and a failure turns M1 off for the whole run. The earlier
PDL-across-the-chain attempt moved d1 by -2.3 / +0.5 us (179cecdac87b4872), so the slice also
needs the per-mode evidence of slices 2-3 before it is worth a check.

## Queued beside M-B: MoE mode 78 at 32 tokens (coordinator, 2026-10-31)

Mode 78 (51 + paired-row scale decode + the fin4 final pass below 64 tokens, muse_moe_w4a16 at
d73c4fd) was 1.5-3.5 us under 51 at 32 tokens on 35 of 36 paired seeds and failed its gate at 128
std (docs/runs-moe.md, "Donor MoE (M-A) against mode 51"). Port: `tools/dl_moe_inline.py` now
takes a target file and builds `LAYER_MAIN_EXTRA = ((78, 64),)`, writing
`_MOE1_MAIN_MAX_M = {'main-lean-grp-fin4ps': 64}` and a tuner filter, so 78 is offered at 32
tokens only. ybuf check: the layer allocates ybuf as `(_MOE1_EMAX, D_EXPERT)` =
(512 x 8, 4096) = (4096, 4096) FP32, the device's kEmax x kH; 78's partials use rows 4064-4095
and only below 64 tokens, where the pair rows end under 63 x 8 = 504 (the source's static_assert
states the same bound); nothing else in the layer uses ybuf. So the last 32 rows are free.

Pre-registered rule (written before any check of it): 78 ships only if (1) the stable MoE
tuner picks it at 32 tokens over the preferred 51 in both cand prepares (a saving above
max(0.3 us, 2 x MAD) in at least 8 of 9 rounds on its four routings), and (2) a four-arm bench
on fresh seeds (280001+) against the incumbent reads, on the d32 load, all four cross ratios
above 1 and the pooled d32 ratio above 1 + 2 x the larger d32 null, with no other load's pooled
ratio below 1 - 2 x its larger null. The whole-score decision rule is not used for it alone:
one load out of six moves, about 0.1-0.2% of the score.

## Trigger-order invariant (applies to every new trigger)

Several kernels read, before their own wait, data written two or more launches back, and rely
on that writer being complete. In the source: `boundary_kernel` (the FFN boundary at T <= 8)
loads the stream written by the attention boundary before its wait on the unpooler, citing
"every path from it to here passes GEMM1 or Marlin, which wait before they trigger"; the audit
table in docs/runs-decode-layer.md ("PDL provenance of every load before a
griddepcontrol.wait") rests on "every kernel completes only after its own wait". A secondary
can launch once every CTA of its primary has triggered, so with triggers placed before waits a
chain of early triggers could start a reader while a writer two launches back still runs.

Rule for M-B: for each pre-wait read of a writer two or more launches back, the chain between
them must contain a kernel that waits before it triggers. The MoE's new trigger follows its
own wait (the grid barrier comes after it). The sk entry trigger precedes its wait, so its
successors were checked one by one (layer_mb2s, all switches on):

| sk launch -> successor | successor's pre-wait reads | safe because |
|---|---|---|
| qkv -> attention (attn) | K/V cache rows below `seq_lens`, page table, `seq_lens` | harness inputs, written before the step |
| o -> attention boundary | the step's input stream, alpha, beta, the next norm weight | harness input and prepare-time state |
| skf front -> route | the pre-expert norm weight, the router bias | prepare-time state |
| unpooler -> FFN boundary (T <= 8) | the stream written by the attention boundary in this step, gate and norm weights | the unpooler launches only after the MoE triggered, which is after the MoE's wait, so route is complete; route and skf complete only after every one of their CTAs passed its wait, so the attention boundary is complete |

Entry triggers that already existed (`attn_boundary_kernel`, ONE_TRIG in attention) are
unchanged; with attn on, attention launches early behind qkv and ONE_TRIG lets o launch
early in turn, and o reads only weights before its wait.
