# M-C: fewer, fused glue kernels at one token (design, 2026-10-31; not built)

Diagram: [plan-2026-10-31-m-c-t1-fusion.html](plan-2026-10-31-m-c-t1-fusion.html).

## Why (measured), and the hypothesis it tests

M-B (docs/plan-2026-10-31-m-b-pdl-chain.md) found that PDL on our own kernels saves at most
0.8 us at 8 tokens and nothing at 16 / 32, while the donor's layer (alygg77/okc-muse-spark-kernels
@ 6cae49a, read in the scratchpad only; every line of M-C would be our own code) overlaps 7-10 us
of kernel time at d1s / v1s / d16s / d32s (F-LAYER-0, checks 20abe6291eb04ed6 and
00607fc6f8c84c77). The hypothesis from M-B: the donor's lead at small token counts comes from
fewer, fused glue kernels, not from PDL itself. M-C tests it where the kernel count differs most,
one token.

Decode 1 sliding, per-kernel medians from the F-LAYER-0 ABK lines (check 00607fc6f8c84c77, us;
the donor's kernel roles are inferred from their order and size, not from its source):

| stage | ours | donor |
|---|---|---|
| qkv projection | cgemv40 qkv 10.8 | 11.3 |
| attention | attn_one 5.0 | 5.9 |
| o projection + attention boundary (residual, two norms, FP32 router input) | cgemv o 5.5 + boundary 3.9 = 9.4 | one kernel 7.0 |
| MoE front: pooler, pre-expert norm, router, routing | cgemv pooler 17.0 + pre_norm 1.9 + router 5.1 + routing 4.3 = 28.3 | 20.4 + 3.0 = 23.4 |
| MoE | sk8w3s 24.9 | 25.3 |
| unpooler + FFN boundary | 3.5 + 3.9 = 7.4 | 4.9 + 2.3 = 7.2 |
| step span | 85.9 (11 kernels) | 74.0 (8 kernels) |

The two fused stages account for about 7 us of the 12 us gap (2.4 at o + boundary, 4.9 at the
front), so they are the test.

## Design

### C-1: the o projection's epilogue forms boundary 1

Today `cgemv<64,1>` writes the BF16 branch (8192 values) and `attn_boundary_kernel` (a cluster of
8 CTAs) reads it with the stream, adds the residual in FP32, takes both sums of squares, and
writes the FP32 stream, the BF16 next input and the FP32 router copy.

Fused: each o CTA (2 output rows per CTA in cgemv) also loads its slice of the input stream and
of the gate / norm weights before its wait, adds the residual for its rows, and publishes its
rows' partial sum of squares. The full-row norm needs every CTA's partial, so the kernel ends
with a last-arriver join: each CTA adds its partial with `red.relaxed.gpu.global.add.f32`, then
`__threadfence()` and `__syncthreads()`, then thread 0 `atomicInc`s a counter; the CTA that
reads count - 1 computes the scale and a second pass over the 8192 values writes the normalized
outputs. This is the projection agent's `WV_GRED` join pattern with its PTX argument (the
fence.sc total order, docs/proj-port-guide.md "Global split-K join"), which needs no
co-residency, unlike a grid barrier over all CTAs. The counter and the accumulator are reset by
the last CTA, so graph replays need no host reset. The second pass costs one more read of 8192
FP32 values by one CTA (about 1 us on one SM); a cheaper form has every CTA spin on a
generation flag after its arrival, which needs co-residency and is rejected for that reason
(coordinator method: progress must never depend on co-residency).

### C-1 revised before building (2026-10-31): a cooperative kernel with two grid barriers

The last-arriver form above does not work as written. The boundary needs two full-row
reductions in sequence: the sum of squares of the BF16 branch (post-attention norm), then the
sum of squares of the new stream, which contains a BF16 rounding of the normalized branch
(`bf16r()` in `attn_boundary_kernel`). After a last-arriver join only one CTA knows the first
scale, so it would have to produce all 8192 outputs alone (one SM streaming about 120 KB, likely
slower than today's 3.9 us boundary). Expanding the second sum of squares into per-CTA partials
(sum s^2 + 2c sum s g b + c^2 sum (g b)^2) avoids that but drops the BF16 rounding the reference
applies before the residual add, so it is not the same computation.

The exact form is a persistent cooperative kernel (cudaLaunchAttributeCooperative, grid at
most the co-resident CTA count, probed under graph capture as `attn_one_probe_coop` does, C-1
off if refused): phase 1 computes the o rows (the cgemv stream format and escape table) and
each CTA's partial sum of squares of its BF16 branch values; grid barrier 1; every CTA sums the
partials in a fixed order (deterministic), takes the first scale, computes the new stream for
its share of the 8192 columns exactly as the boundary kernel does and its partial sum of
squares; grid barrier 2; every CTA forms the second scale and writes its share of the stream,
the BF16 next input and the FP32 router copy. Co-residency is guaranteed by the cooperative
launch, which the attention's cooperative modes already rely on; the barrier is the MoE's
generation counter (atomicAdd after `__threadfence()`, the last arrival bumps the generation,
the others spin with `ld.acquire.gpu`), whose ordering argument is in the MoE port guide. Only
the summation order of the two norms changes against today's path.

### C-1 implementation (2026-10-31; built and compiled, not run on a GPU)

Candidate `muse_decode_layer/cand/layer_mc1s.py`, made from `layer_m1rt51sc1.py` by
`tools/dl_mc_c1_patch.py` (byte-identical on a second run). The kernel lives in its own
extension module (`_MC1_SRC` plus `_SKP_GEMV_SRC` built with `GEMV_DEVICE_ONLY`, so cgemv's
`chunk_fma` is the same source), built on its own thread; every source block of the base file is
unchanged, which the patch tool checks.

Kernel `mc1_kernel`, launched with `cudaLaunchAttributeCooperative` only:

- Geometry: 768 threads (8 units of cgemv<64, 1>'s CTA: two compute warps and the escape warp),
  `__launch_bounds__(768, 2)`; ptxas: 40 registers, no spills, 2016 B static shared memory,
  16 named barriers. The grid is the occupancy limit times the SM count (264 on 132 SMs at two
  CTAs per SM, by threads and registers), checked against [64, 1024] and probed under capture.
- Phase 1: unit u of CTA b takes row pairs `b * 8 + u + i * 8G` (two rounds at G = 264), with
  cgemv_body's loads, `chunk_fma`, warp sums, escape loop and final sum order (zero, warp 0,
  warp 1, escapes), so the BF16 branch is bitwise cgemv's. Two pairs in flight per unit spilled
  at the 40-register limit, so a unit takes one pair per round. The unit's first thread adds the
  squares of the two stored BF16 values to its running sum; the CTA sums its units in order and
  stores `part[b]`.
- Barrier 1, then warp 0 of every CTA sums `part[0..G)` in a fixed order (lane l adds entries
  l, l + 32, ... in order, then the xor butterfly), so every CTA gets the same first scale.
- Phase 2: CTA b takes float4 groups `[b * 2048 / G, (b + 1) * 2048 / G)` (at most 32), one per
  lane of warp 0, and computes `s = alpha * stream + beta * bf16r(f * rb)` and its squares with
  the boundary's expressions. The SASS of both kernels shows the same contraction,
  `fma(alpha, stream, beta * bf16r(f * rb))`, and the same scale argument
  `fma(total, 2^-13, eps)`. The stream, alpha, beta and w_next slices are copied to shared memory
  by `cp.async` at kernel entry, so phase 2 does not wait on DRAM.
- Barrier 2, the second scale the same way, then the stream, the BF16 next input and the FP32
  router copy, `o = s * rn * w_next` as the boundary writes them.

SASS counts (`tools/dl_mc_c1_sass.sh`): mc1_kernel instr 1297, MEMBAR 2 (one MEMBAR.SC.GPU per
barrier), ATOMG 5 (two arrivals, the generation bump, two timeout flags), BAR 7, NANOSLEEP 2,
LDGSTS 4, ACQBULK 0, PREEXIT 0, LDL 0, STL 0.

#### PTX memory-model chain for the two grid barriers

Section numbers as `docs/proj-port-guide.md` cites the PTX ISA 9.4 memory consistency model;
the release and acquire patterns section (8.8) is my reading of that numbering and should be
checked against the ISA text.

Barrier k (counter `ctr[2k + (gen & 1)]`): in every CTA A, the threads' stores (the branch
values and `part[A]` for barrier 1, `part[G_MAX + A]` for barrier 2) are followed by
`__syncthreads()` (`bar.sync 0`), then thread 0 runs `__threadfence()` (`fence.sc.gpu`,
MEMBAR.SC.GPU in the SASS) and `atomicAdd` (`atom.add.relaxed.gpu`, morally strong at .gpu scope, 8.7). The
fence followed by the strong write is a release pattern (8.8). In every CTA B, thread 0 spins on
`ld.acquire.gpu` until it reads G, then `__syncthreads()`, then the threads read with `ld.cg`.

1. The value G that B's acquire reads was written by the last `atom.add` on the counter; each
   CTA's `atom.add` precedes it in the counter's read-modify-write chain (coherence order of the
   location, 8.9.6, the coherence axiom 8.10.1, and atomicity, 8.10.3). So A's release write precedes B's acquire read in
   observation order, and A's release pattern synchronizes with B's acquire (8.9.4).
2. Base causality order (8.9.5) then runs: A's stores, program order and `bar.sync` (a CTA-scope
   synchronization) to A's thread 0, its fence and atom, synchronizes-with to B's acquire, then
   `bar.sync` to B's reading threads. Each B load is preceded in causality order by each A store
   to the same location, so it cannot read an older value (causality axiom, 8.10.6).
3. The same chain needs no `fence.sc` total order (8.9.3) beyond what the release pattern gives;
   `fence.sc` is kept because it is the documented MoE pattern and stronger than
   `fence.acq_rel`.

Counter and generation reuse:

- Every CTA reads `ctr[0]` (the generation g) at entry, before its barrier-1 arrival. The bump
  to g + 1 is made by the last arrival at barrier 2, whose RMW observes every CTA's barrier-2
  arrival, which each CTA makes after its barrier-1 wait, which comes after its read of g in
  program order. If a read returned g + 1 the bump would precede the read in causality order
  and follow it as well, which the causality axiom forbids. Review fix (2026-10-31): the
  arrival atomics are relaxed, so without a fence the bump's ordering after the arrivals rests
  on the control dependency on `old == G - 1` only; the last arrival now executes
  `__threadfence()` before the bump, which makes it an acquire of the RMW chain.
- The SUMMARY's `mc1 ok=... on=... timeout flags N` reads `ctr[1]` after the timeline, so a
  timeout in any of prepare's C-1 launches is visible in the log tail (review should-fix).
- Launch g uses the counters of parity g & 1; CTA 0 of launch g zeroes the parity (g + 1) & 1
  counters at entry. Those were launch g - 1's, which completed before launch g started (stream
  order: launch g carries no PDL attribute). The zeros are ordered before launch g + 1, which
  uses them, by launch g's completion. No host reset is needed between graph replays.
- A wait past 2^31 cycles sets `ctr[1]` and gives up (the output is then wrong). The self-test
  and the gate read it, and the gate also checks that the generation advanced by exactly the
  number of C-1 launches (two warm-ups plus 2000 replays) and that this generation's counters
  are zero and the previous one's are G.

#### Launch ordering against the rest of the step (coordinator condition, 2026-10-31)

- The C-1 source contains no `griddepcontrol` (no wait, no `launch_dependents`), and its launch
  sets only `cudaLaunchAttributeCooperative`. The patch tool fails if either changes, the host
  wrapper checks the attribute list before the launch, and the SASS has ACQBULK 0 and PREEXIT 0.
- Predecessor: the attention kernel. C-1 is not its PDL secondary, so C-1 starts after the
  attention grid has completed and its writes are visible (stream order).
- Successors at one token, on the main stream: the front variant picked at T = 1 (preferred
  "fp32"). Under "fp32" the pooler runs as `_proj(..., "pooler")`: the cgemv pick launches with
  `<<<>>>` (no attribute); if there is no sk pick and the mma projection serves it, `proj_fwd`
  launches with the PDL attribute (`launch_ex_pdl`). Under "skf" / "skf-r" the fused pooler and
  router launches with the PDL bit when M1's mode at bucket 8 is "o+front". On the side stream,
  forked by an event right after C-1, the FP32 router (`router_launch`, `kPdlGlue`) carries the
  PDL attribute. So some successors can be PDL secondaries of C-1.
- Why C-1's barriers cannot wait on them: a PDL secondary may launch once every CTA of the
  primary has executed `griddepcontrol.launch_dependents` or exited. C-1 never executes the
  trigger, so the implicit trigger is each CTA's exit, and no successor CTA can be launched while
  any C-1 CTA is still running. Every C-1 CTA passes both barriers before it exits, and the
  cooperative launch guarantees that all G CTAs are resident together, so nothing a barrier waits
  for can be kept off an SM by a successor.

#### Validation as built

- Self-test (`_mc1_self_test`, 8 inputs: o input scales 0.01 to 2, stream scales 0.3 to 30, and
  a one-spike input): against `_proj` (the cgemv pick) then our 8-CTA boundary. The branch must
  be bitwise equal. The stream may differ only on elements whose normalized-branch BF16 rounding
  flipped, which happens per branch mantissa (`bf16r(f * rb)` depends only on f's 8-bit mantissa
  once rb is fixed, so a 1-ulp change of rb flips all elements of one mantissa class, about 64 of
  8192): at most 2 mantissa classes, each element within beta x 2^-7 x |f rb| plus 8 FP32 ulps.
  Elsewhere the stream is bitwise equal, the router copy is within 8 ulps and relative L2 1e-6,
  and the BF16 next input differs by one ulp on at most 8 elements. The counters must be clean
  and the generation must advance once per call. This is looser than the brief's "relative L2
  under 1e-6 on the stream" because a mantissa-class flip gives about 5e-4 relative L2 on the
  stream (CPU emulation below); the bound per flipped element is what keeps the check tight.
- Fresh-input gate (`_mc1_gate`): `_m1_stress`'s method at 1x1 sliding and full: five (stream,
  attn_input) pairs, C-1-off eager references, C-1 off (control) and on (candidate) captured and
  replayed 2000 times each with fresh inputs, `_gate_verdict`, the counter check, and the
  negative controls on an eager C-1-on output. A failure raises with its classes and lands in
  SUMMARY's failed list through `_try`.
- Pick (`_mc1_tune`): whole-step graphs at 1x1 sliding and full, off and on, 9 rotated rounds of
  cold-L2 medians (`_m1_rounds`), load 46 x sliding + 16 x full over 62, `_stable_pick` with
  "off" preferred; the note goes to `stable_report["mc1 T=1"]` and the bench's `ABPOL mc1`.
- CPU emulation (`tools/dl_mc_c1_cpu_check.py`, both summation orders in FP32 with the SASS's
  contractions; rsqrtf modelled as correctly rounded in both): 1000 inputs at G = 264, 200 at
  G = 132 and 200 at G = 1024. At G = 264 the first scale differed in 241 of 1000 inputs; no
  stream flip occurred; the next input differed on at most 2 elements per input (27 in 1000); the
  router copy was within 4 ulps and 1.4e-7 relative L2. With rb forced one ulp up, 221 elements
  flipped over the 1000 inputs, always within one mantissa class per input (up to 64 elements,
  4.9e-4 relative L2 on the stream, each element at most 0.87 of the flip bound). The row-pair
  and column partitions cover every item once for every grid from 64 to 1024.

Risk: the gate's bank holds five fixed pairs per shape. If one pair's first scale lands on a
mantissa-class flip (about 1.4% of inputs whose scale differs, so roughly 3% per prepare over
both shapes, estimated from the emulation), every replay of that pair differs from the C-1-off
reference by about 5e-4 on the stream, over GATE_BOUND, and the gate turns C-1 off. That fails
closed.

Expected saving (hypothesis, not measured): the o cgemv streams 12.6 MB (5.5 us measured at d1s,
3.1 us at 4 TB/s). Phase 1 keeps about 48 KB in flight per SM against cgemv's 64 KB, so it
should take about the cgemv's time; each barrier costs an atomic chain of 264 arrivals on one
address and an acquire round trip, about 0.5 to 1 us; phase 2 reads L2-resident data. About
5 + 2 x 0.8 + 0.5 = 7 us against 9.4 us, so 1.5 to 2.5 us at d1s, near the donor's 7.0 us. The
stop rule above (under 1 us closes it) decides.

### C-2: the one-token front in one launch

The FP32 front at one token runs the pooler as `cgemv<512,1>`, then `pre_norm`, the FP32 router
and `routing_kernel`. Fused: one kernel whose CTAs compute the pooler rows (as cgemv) and the
router logits (the router's 256 rows as extra CTAs, FP32 input), with the pre-expert norm of the
pooled row and the top-8 routing in the last-arriving CTA (the same join as C-1), which then
writes the MoE's metadata. The MoE launches as its PDL secondary, as it does behind route today.

### Ordering, including the donor's E2 pattern

The donor's `_gle_route` reads the o kernel's outputs before its `griddepcontrol.wait` while its
primary is the pooler (finding E2 of the donor audit). That is a two-back read whose safety rests
on every kernel between them waiting before it triggers. M-C does not copy it:

- C-1 reads before its wait only weights, the gate / norm parameters and the step's input
  stream (harness input); the branch it adds is its own output.
- C-2 reads the boundary's outputs (the FP32 router copy and the BF16 next input) only after its
  wait; its primary is C-1, so the wait alone makes them visible.
- Neither kernel triggers its dependents before its own wait, so the chain property that the
  FFN boundary's pre-wait read relies on (docs/plan-2026-10-31-m-b-pdl-chain.md, trigger-order
  rule) holds. The last-arriver join's ordering is the projection agent's argument cited above.

### Validation

Bitwise equality is not expected (the join changes the summation order of the norms). Gates:
FP32 references of the boundary outputs and the router logits within the existing self-test
bounds, routing ids equal to the separate path's on 100 judge-like inputs (or within the
route-variant check's rule), the fresh-input replay gate on the fused front + MoE graph, and a
clean counter after 2000 replays. SASS: fence, membar and atomic counts listed per kernel.

## Cost, budget and stopping rule

Two new kernels and their host wrappers. A first slice (C-1 alone, behind a switch timed by the
stable tuner against the two-kernel path) is about a day of work. Decision: an in-process paired
saving of at least 1.5 us at d1 sliding (9 of 9 rounds) for C-1, then a four-arm bench with
independent prepares. Stop if C-1 saves under 1 us: then the hypothesis that fusion explains the
donor's lead is rejected at this stage too, and the remaining gap is per-kernel speed.

## Gaps

- The donor's kernel roles at d1 are inferred from order and duration; its exact fusion
  boundaries were not read for this plan.
- Whether the pooler (512 rows per CTA group in cgemv) and the router (FP32, 256 rows) can share
  one launch without slowing the pooler's memory stream is unmeasured.

## Result (2026-10-31, check 5ae9ce318e014edf)

C-1 passed its cooperative probe, self-test and fresh-input gate on the judge's GPU and was
then timed by its stable pick against the two-kernel path: the whole 1x1 step was 1.16 us
slower with C-1 on, in all 9 rounds. The stopping rule (stop under 1 us of saving) is met; C-1
is in docs/ruled-out.md and C-2 is not built. Together with M-B, both forms of glue overlap
tested on our own kernels (PDL chaining and fusion behind grid barriers) lost or gained under
1 us at the stage where the donor gains most, so the remaining gap at d1 is more likely the
speed of individual kernels (the one-token front above all: 28.3 us summed against 23.4).

## Reframe: the one-token front per kernel (2026-10-31, after three weak structural results)

The 28.3 us front in the table above came from F-LAYER-0 (00607fc6f8c84c77), whose "ours" arm was
the release (b9c23bda), where the four front kernels ran one after another (kernel sum 85.7 us
equal to the step span 85.9). The current incumbent runs them differently. Per-kernel medians
at decode 1 sliding, us (layer_m1rt51s from bench b70cfa48c0264d20's inc arm, kernel names from
its ABKN lines; donor and release from 00607fc6f8c84c77, roles of the donor's kernels inferred
from launch order; different benches have different inputs, so compare within a column pair
only where noted):

| stage | release (00607) | donor (00607, same inputs) | layer_m1rt51s (b70cfa48) |
|---|---|---|---|
| qkv | 10.8 | 11.3 | 10.5 |
| attention | 5.0 | 5.9 | 5.0 |
| o | 5.5 | 7.0 (o + boundary fused) | 5.4 |
| attention boundary | 3.9 | (in the above) | 6.0 (PDL secondary: includes its wait) |
| router (FP32, side stream) | 5.1 | 20.4 (front, fused) | 9.1 (launched first, overlaps the pooler) |
| routing (side stream) | 4.3 | (in the above) | 10.6 (PDL secondary of the router: includes its wait) |
| pooler (cgemv 512) | 17.0 | (in the above) | 17.2 |
| pre-expert norm | 1.9 | 3.0 | 1.5 |
| MoE | 24.9 | 25.3 | 32.0 (other inputs) |
| unpooler + FFN boundary | 7.4 | 7.2 | 7.3 |
| step span / kernel sum | 85.9 / 85.7 | 74.0 / 80.0 | 87.9 / 104.6 |

Readings:

- In the incumbent the router and routing overlap the pooler (kernel sum 16.7 us above the
  span), so its main-stream front is the pooler plus the pre-expert norm, 18.7 us, against the
  donor's 23.4. The pooler streams 8192 x 4096 12-bit weights, about 50 MB, whose time at the
  H200's 3.35 TB/s is about 15 us, so it is within about 2 us of its floor. The front is no
  longer where the donor gains.
- The F-LAYER-0 gap (11.9 us at d1s) was measured against the release, not the incumbent. On
  the stages that line up, the release and the donor are within 1 us of each other except the
  front (serial in the release) and the o + boundary pair (2.4 us). Whether the incumbent still
  trails the donor at d1, and where, needs a same-input bench of the incumbent against the
  donor with kernel names printed; the per-kernel attack has no target until then.
- Proposed measurement (one check, a bench with the donor as the public arm, scratchpad only,
  never submitted): cand = layer_m1rt51sc1g, loads d1 and v1 only so the ABKN map and every
  ABK line survive the tail. The largest same-input single-kernel gap it shows is the target
  for the A/B.
