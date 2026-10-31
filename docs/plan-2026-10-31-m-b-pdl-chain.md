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

The plan's diagram (user rule "plans always in md" asks for one) is not drawn yet.
