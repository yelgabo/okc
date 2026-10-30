# Porting the fused MoE kernel into muse_decode_layer

The routed-experts block of `muse_moe_w4a16/submission.py` is one CUDA launch per call. This
guide gives what a layer needs to drop it in: the tensors, the scratch, the prepare steps, the
launch, and the points where it rounds to bf16. The kernel source is `_CUDA_SRC` in that
file; the host entry is `moe_run`.

## What one launch does

Grid = SM count (132 on H200), one CTA per SM, no cooperative launch (the shared-memory
footprint guarantees co-residency). Each CTA:

1. Routing prologue, run redundantly by every CTA from `topk_ids`: local expert id per entry
   (`id - local_start` when in `[0, local_count)`, else -1), a histogram, pair-row offsets
   (expert-major; entry i gets pair row `pos[i]`), and a job list.
2. Phase A: GEMM1 over the job's tokens for the job's expert, SwiGLU applied in the epilogue.
3. Phase B: GEMM2 over the SwiGLU output, writing FP32 `y * (1 + post_norm_w)` to `ybuf` and
   accumulating each pair row's sum of squares into `ss`.
4. A software grid barrier (atomic counter, generation-indexed).
5. The final pass: `out[t] = sum over local slots s of topk_w[t, s] * rsqrt(ss[q] / 4096 + eps)
   * ybuf[q]` with `q = pos[8 t + s]`, rounded to bf16. Tokens with no local slot get zeros.

Two kernels share this structure and the routing code:

- `moe_fused_kernel<SPLIT>` (288 threads: 8 compute warps and a helper warp) for M <= 16.
  Jobs are up to 8 tokens of one expert (an expert with more tokens becomes several jobs that
  re-stream its weights). A unit is 16 weight rows over the full K, one 36,864-byte bulk copy
  into a 4-slot ring; the 8 warps split K and use `mma.sync m16n8k16` with the job's x rows in
  registers; the helper warp sums the warps' partials and runs the epilogue.
- `moe_main_kernel` (576 threads: 4 consumer warpgroups, a weights producer thread and an x
  producer thread) for M >= 17. Jobs are up to 64 tokens; a unit is 256 rows streamed in 16
  stages of 256 k; consumers run register-A `wgmma m64nNk16` (N = 8 x ceil(tokens / 8)) with
  the stage's x in shared memory; stages are split over CTAs by a cost table (stream-K) and
  units cut between CTAs are summed through a global fixup slot. The weights ring has 5, 4 or
  3 slots of 36 KB next to 3 x slots of 8, 16 or 32 KB, chosen per launch from the largest
  job's N (every CTA derives the same sizes in its routing prologue).

The skinny kernel is built in several variants (`SKINNY_VARIANTS`: 8 or 16 compute warps, 3
or 4 ring slots, and loads-only diagnostic modes that stream the weights without computing;
since 2026-10-29 `(8, 3, 0)`, sk8w3s, the 3-slot ring, which beat sk8w4s on 36 of 36 paired
seeds at 1, 8 and 16 tokens in 75bca9775f6f4b0b; the inliner carries it unchanged, see
docs/runs-moe.md, "Skinny weight stream against the public kernel", Port);
the fat kernel in a computing and a loads-only mode. `prepare` times every computing variant
per batch size (graph replay, cold L2, synthetic routings) and keeps the fastest as
`state["policy"]`, keyed by the sizes in `POLICY_BS`; `kernel_fn` looks the variant up by M.
Without a policy entry, M >= `MAIN_MIN_M` (17) runs the fat kernel when `state["main_ok"]`
holds (the self-test clears it if the fat path fails), else the split skinny path, which
handles every size. The previous kernel (per-warp register-path mma.sync GEMMs, check
a49578e5c30348bb) is kept as `_run_legacy` with its tuned configs for 32, 128 and 256 tokens
(`LEGACY_POLICIES`, variant ids from `VARIANT_LEGACY0`); it is a bench candidate from 16
tokens up and needs its own weight layout (`_legacy_prepare`, 906 MB) and two more
extensions. Once the fused kernel wins everywhere it can be dropped.

## Tensors the layer must pass

Inputs per call (all on the same device, contiguous):

| name | dtype, shape | notes |
|---|---|---|
| `x` | bf16 [M, 4096] | the MoE block's input rows |
| `topk_ids` | int32 [M, 8] | global expert ids; 16-byte aligned (torch allocations are) |
| `topk_w` | float32 [M, 8] | routing weights as given |
| `out` | bf16 [M, 4096] | written in full |

Weights and constants prepared once (`prepare`), per local expert range `[lo, lo + nloc)`,
`nloc <= 32`:

| name | dtype, shape | made by |
|---|---|---|
| `w13` | uint8 [nloc, 512, 36864] | `repack_expert(packed, sf, 1)` per expert: 16-row units, 8 gate rows then their 8 up rows |
| `w2` | uint8 [nloc, 256, 36864] | `repack_expert(packed, sf, 2)` |
| `w13f` | uint8 [nloc, 32, 16, 36864] | `repack_expert_fat(packed, sf, 1)`: 256-row units x 16 stages |
| `w2f` | uint8 [nloc, 16, 16, 36864] | `repack_expert_fat(packed, sf, 2)` |
| `gs13`, `gs2` | float32 [nloc] | the global scales, as given |
| `nw` | float32 [4096] | `1 + post_norm_w` in FP32 |

The repacks permute bits and bytes only; the values stay NVFP4 (E2M1 codes and E4M3 block
scales). `scales_ok` must hold (positive normal E4M3 scales), which the challenge's weights
satisfy; otherwise the dequant's PRMT + IMAD scale trick is inexact and the kernel must not be
used. The two repacks hold 906 MB each for 32 experts.

Scratch allocated once (`_scratch`), reused by every launch; nothing is reset between
launches from the host:

| name | dtype, shape | role |
|---|---|---|
| `inter` | bf16 [4096, 4096] | SwiGLU output by pair row (skinny path) |
| `ybuf` | float32 [4096, 4096] | GEMM2 output times (1 + w), by pair row |
| `ss` | float32 [2, 4096] | sums of squares, double-buffered by the launch generation |
| `ctr` | int32 [64] | ctr[3] generation, ctr[4], ctr[5] the two barrier counters, ctr[8] spin-timeout flags, ctr[10] mode 23's phase B claim counter (zeroed by CTA 0 after the grid barrier) |
| `done_a` | int32 [544] | per job, phase A units finished (skinny path) |
| `xa`, `xb` | uint8 [(512 + 96) x 65536] | fat path: x rows and SwiGLU output in the per-stage wgmma B layout |
| `fix` | float32 [2 x SMs x 4 x 128 x 32] | fat path: fixup slots for units cut between CTAs |
| `fixcnt` | int32 [2 x SMs x 4] | their stage counts |
| `xready` | int32 [96] | fat path: gathered x rows per job |
| `done_as` | int32 [96 x 16] | fat path: finished phase A units per (job, phase B stage) |

Sizes assume at most 512 tokens per launch (`CHUNK_TOKENS`); `moe_run` splits larger M into
launches of 512. Every per-launch flag is reset inside the launch (after the grid barrier, or
by the finisher that used it), and the barrier counter and `ss` copy alternate with the
generation, so graph replays need no host-side memset.

## Prepare steps

1. `_repack(weights, lo, nloc)`: both layouts and the global scales, about 1 s on GPU.
2. `nw = (post_norm_w.float() + 1)`.
3. `_scratch(device)`.
4. Build the extension once (`_ext()`, one `load_inline`, about 40 s; the fat path's wgmma
   wrappers are generated by `_wgmma_code()`).
5. Self-test against the FP32 reference: every computing skinny variant at M = 1, 4, 8, 16
   and the fat path at 32, 33, 128, 257, normal and skewed routings; the tolerance is 8e-3
   relative L2 (measured 3.8e-3 to 4.6e-3; the judge allows 1.5e-2). A spin timeout in any
   launch (ctr[8] != 0) fails the test. A failing fat path sets `main_ok = False`.
6. The variant bench (`_bench`) and the policy, then the policy's variants once more at every
   test size.

The judge gives `prepare` 300 s; the whole of the above takes about 60 to 90 s (the build is
the largest part).

## The launch

`moe_run(x, topk_ids, topk_w, w13, w2, gs13, gs2, nw, inter, ybuf, ss, ctr, done_a, out,
w13f, w2f, xa, xb, fix, fixcnt, xready, done_as, lo, nloc, eps, variant, cost)` with `variant`
an index into `SKINNY_VARIANTS` (skinny, split jobs of 8 tokens) or `VARIANT_MAIN0 + i` with
`MAIN_MODES[i]` the fat mode, and `cost` either `[]` (the compiled stream-K stage costs) or 18
ints, `cost[9 * phase + ntt]` for ntt 1 to 8 (entries 0 and 9 unused). Since 2026-10-28 (commit
068b3ce) the cost argument is required; a layer port made from an earlier file passes `[]`.
Fat modes: 0 wgmma for every job; 6 jobs of up to 2 n8 tiles on mma.sync, the rest on wgmma;
12, 13, 14 are 6, 0 and 6 with fence.acq_rel.gpu in place of fence.sc.gpu in the epilogue and
fixup, 14 also with two mma.sync accumulator chains; 1 streams only; 11 is 6 with clock stamps
written to `inter` (never a policy choice). `VARIANT_FIT0 + i` in Python runs `VARIANT_MAIN0 +
i` with the costs fitted from the stamps in prepare (`state["fit_cost"]`).
Since 2026-10-29: 16 is 14 with phase B boundaries on unit boundaries; 20 ("main-own") is 16
with every phase A unit owned whole by one CTA (no phase A cut or fixup; units dealt in snake
order by n-tile count), 2-4 us under 16 at 128 and 256 tokens in three same-process A/Bs and
the standalone policy at both sizes. Porting 20 to the layer means adding it to
`tools/dl_moe_inline.py`'s `LAYER_MODES`; the routing prologue gains a counting sort of the
jobs (S.ord) and SmemMain grows by 388 bytes (228,480 in all). 17, 18, 19, 21 and the staged
pair are timing-only or lost (docs/runs-moe.md, "Staged GEMM1 gate").
Also since 2026-10-29: 23 ("main-own-dynb") is 20 with phase B units claimed at run time: each
CTA's weights producer takes the next unit (S.ord order, 16 per job) from ctr[10] when its ring
frees a slot and passes it to the x producer and consumers through S.bq; both phases run largest
n-tile count first. 16-23 us under 20 at 128 and 256 tokens on 12 of 12 paired seeds per size
in two checks (docs/runs-moe.md, "Phase B claimed at run time") and the standalone policy at both
sizes; SmemMain is 228,736 bytes. Porting it means adding 23 to `LAYER_MODES`. 22 and 24 are 20
and 23 with stamps (timing only).
Also since 2026-10-29: 26 ("main-dynb-fs") is 23 with the routes counted once by every warp (the
routing warp sums the per-warp counts, the positions reuse them), a batched x gather (a CTA's 16-byte
row loads before its stores) and no stage-range searches (owned and claimed units read none). 4-10
us under 23 at 128 and 256 tokens on 66 of 66 paired seeds (standalone, held-out, skewed and
layer-like routing; docs/runs-moe.md, "Same-process A/B against the public MoE") and the standalone
policy at both sizes in 817eab6401d54a84 (2.104x). Same interface, scratch and SmemMain size as 23;
porting it means adding 26 to `LAYER_MODES`. 25 and 27 are 23 and 26 with thread-0 phase stamps
(timing only; they write 16 int64 per CTA to `inter`).
Also since 2026-10-29: 28 ("main-lean") is 26 with a leaner consumer stage (ring size and x slot
base in registers, the weights slot released by every consumer thread with its own
fence.proxy.async.shared::cta and arrive behind the stage's first wgmma group, so `wempty` and
`xempty` count kNC x 32 arrivals) and a one-pass S.ord sort. Means of 6.4 to 12.5 us under 26 at 32,
128 and 256 tokens, 72 of 72 paired seeds lower (layer-like, standalone, held-out and skewed routing,
three checks; docs/runs-moe.md,
"Consumer stage and the ring slot releases"); standalone policy at 128 and 256 in f928f8a802014e4f
(2.149x) and its repeat 13fe68e7ca2e423f (2.158x). Same interface, scratch and SmemMain size as 26.
Porting it: add 28 to `LAYER_MODES` and `28: "main-lean"` to `MAIN_NAMES` in
`tools/dl_moe_inline.py`. The standalone `MAIN_MODES` is now (0, 16, 26, 28), and the inliner keeps
only modes listed there, so a `LAYER_MODES` that still names 20 or 23 loses them silently. On layer
routing at 32 tokens 28 is under the skinny sk8w4s by 18 us (6 of 6 seeds), so a layer bench on
judge-like routing may choose it at decode 32 too. 29 is 28 with the stamps (timing only).
Also since 2026-10-29: 37 ("main-lean-silu-bar") is 28 with a lighter phase A epilogue: SiLU's
reciprocal as one `rcp.approx.ftz` (MUFU.RCP) instead of `__frcp_rn` (a range check and a
subroutine per element), and no per-thread `fence.acq_rel.gpu` before the warpgroup's bar.sync
and red.release on done_as (each writer keeps its `fence.proxy.async.global`; the PTX chain is in
docs/runs-moe.md, "Ordering of the phase A publish without the per-thread fence_gpu"). Paired
against 28 on layer routing: -1.9 / -2.1 / -3.8 and -1.7 / -3.4 / -4.2 us at 32 / 128 / 256 tokens
(two checks, 12 of 12 seeds lower at 128 and 256); held-out standalone -3.2 / -4.9, skewed -1.5 /
-2.6; standalone policy at 128 and 256 in 987d024620e44eb7. Outputs move by the approximate
reciprocal (1e-4 against 28, 3.9e-3 against FP32 as before). Same interface, scratch and SmemMain
size as 28. Porting it: add 37 to `LAYER_MODES` and `37: "main-lean-silu-bar"` to `MAIN_NAMES` in
`tools/dl_moe_inline.py` (the standalone `MAIN_MODES` is now (0, 16, 26, 28, 37)). 31 to 36 and
38 to 40 are timing-only or lost (docs/runs-moe.md, "Phase A against the public kernel"); 41 and 42
(37 with the repaired wait counters, and with the light stamps) are timing only (docs/runs-moe.md,
"Fat-path waits with the first try timed").

Since 2026-10-30: 47 ("main-lean-grp-nohoist") is 37 with the weight codes loaded one 4-step
group at a time (one `uint4` per group, the next group loaded at the start of the current one, a
scale word per pair of groups), so a consumer warp holds 10 code and scale registers instead of
18; the waits stay at the stage start and the weights slot is released per thread with its own
`fence.proxy.async.shared::cta` behind the sixth wgmma group (the PTX chain is in docs/runs-moe.md,
"Code groups with the next stage hoisted"). Spills fall from 156 / 416 B to 4 / 88 B. Its
pre-registered gate against 37 (check 5f381b8b6946492d, held-out seeds 13-18): -3.7 / -3.4 / -2.8
us at 256 tokens on layer, standalone and skewed routing (18 of 18 seeds lower), -1.0 to -2.8 at 32
and 128; outputs within 4.4e-5 of 37. Same interface, scratch and SmemMain size as 37. The
standalone `MAIN_MODES` is now (0, 16, 26, 28, 37, 47).

Porting 47 into the layer (on a layer branch that already carries 37, such as
worktree-agent-ae0a55834da6c04d4 at c73ba2a):

1. Bring `muse_moe_w4a16/submission.py` to the commit that has 47 in `MAIN_MODES` and
   `MAIN_CANDIDATES` (ec565a7 or later). The inliner keeps only modes listed in both.
2. In `tools/dl_moe_inline.py`: `LAYER_MODES = (0, 16, 26, 28, 37, 47)` and add
   `47: "main-lean-grp-nohoist"` to `MAIN_NAMES`, with a comment line citing check
   5f381b8b6946492d.
3. Run `python3 tools/dl_moe_inline.py` twice; the second run must leave
   `muse_decode_layer/submission.py` byte-identical. The variant list it prints ends in
   `'main-lean-silu-bar', 'main-lean-grp-nohoist'`.
4. `sh tools/dl_moe_compile_check.sh muse_decode_layer/submission.py` (run it from a folder under
   the home directory: Colima does not mount /private/tmp): `moe1::moe_main_kernel<47>` at 96
   registers, 8 B frame, 4 / 88 B spills, no C7510-C7515.
5. `sh tools/dl_moe_sass_counts.sh muse_decode_layer/submission.py`: kernel 47 with UBLKPF 1 and
   ACQBULK 1 (the PDL preamble), MEMBAR 43, LDL 22, STL 1, one FENCE.VIEW.ASYNC.S more than the
   standalone 47 (the preamble's `fence.mbarrier_init`); the other kernels' counts unchanged from
   the layer copy before the port.
6. The layer's prepare bench on judge routing chooses among the fat modes; no host change is
   needed (47 uses 37's `Params`, scratch and launch path). Judge it with the layer bench
   against the current layer incumbent before a scored run.

Steps 2-5 were run on a scratch copy of the layer branch on 2026-10-30 and passed
(muse_moe_w4a16/profiles/sass_mode47.2026-10-30.txt); the layer files were not changed.

Since 2026-10-30 (later): 51 ("main-lean-grp-finss64") is 47 with, from 64 tokens up, each
route's sum of squares formed in the final pass from its finished ybuf row (one token per item,
two reads of the row, a reduction through the idle rings on named barrier 8) instead of by
shuffles and FP32 atomics in the phase B epilogue; below 64 tokens it runs 47's code. Its
pre-registered gate against 47 (check c6b0e500c3a84c9a, held-out seeds 29-34): -3.9 / -3.8 / -4.0
us at 128 tokens and -3.9 / -4.4 / -7.7 at 256 on layer, standalone and skewed routing, +0.0 /
-1.0 / -0.1 at 32; outputs within 4.6e-5 of 47. Same interface, scratch and SmemMain size as 47;
`ss` is still zeroed for the next launch. 50 (the same at every size) lost at 32 tokens and is
not built. The standalone `MAIN_MODES` is now (0, 16, 26, 28, 37, 47, 51).

Porting 51 into the layer (on origin/main at d96d3d0 or later, which carries 47):

1. Bring `muse_moe_w4a16/submission.py` to the commit that has 51 in `MAIN_MODES` and
   `MAIN_CANDIDATES` (6bbb252 on branch worktree-agent-ad7a25ed576556191).
2. In `tools/dl_moe_inline.py`: `LAYER_MODES = (0, 16, 26, 28, 37, 47, 51)` and add
   `51: "main-lean-grp-finss64"` to `MAIN_NAMES`, with a comment citing check c6b0e500c3a84c9a.
3. Run `python3 tools/dl_moe_inline.py` twice; the second run leaves
   `muse_decode_layer/submission.py` byte-identical. The variant list ends in
   `'main-lean-grp-nohoist', 'main-lean-grp-finss64'`.
4. `sh tools/dl_moe_compile_check.sh muse_decode_layer/submission.py` (from a folder under the
   home directory): `moe1::moe_main_kernel<51>` at 96 registers, no stack frame, no spills, no
   C7510-C7515.
5. `sh tools/dl_moe_sass_counts.sh muse_decode_layer/submission.py`: kernel 51 with instr 17268,
   FENCE.S 22, FENCE.G 11, MEMBAR 43, SYNCS 202, BAR 16 (47's 15 plus the final pass's named
   barrier), ATOMG 148, REDG 9, UBLKPF 1, ACQBULK 1, LDL 0, STL 0; kernel 47 unchanged (instr
   16354, MEMBAR 43, LDL 22, STL 1).
6. The layer's prepare bench on judge routing chooses among the fat modes; no host change.

Steps 2-5 were run on 2026-10-30 on a scratch copy of origin/main (d96d3d0) with this source and
passed; the layer files were not changed.

`_vname(variant)` gives the name the bench prints. It launches on the current stream, is graph-capturable (no
host syncs, no allocation, no value-dependent branching on the host), and needs the whole GPU:
the grid barrier spins until all 132 CTAs arrive, so nothing else may occupy SMs for the
launch's duration, and a PDL successor that starts early would share SMs with the spinning
CTAs. In the layer, launch it alone on the main stream, with the router and the pre-expert
norm complete; the layer's own `topk_ids` and `topk_weights` tensors pass straight through.

A wait that lasts longer than 2^31 cycles (about a second) marks ctr[8] and gives up instead
of hanging; the result is then wrong. Read ctr[8] after a self-test, never inside the timed
path.

Both kernels begin with `griddepcontrol.wait`, so a launch with the programmatic dependent
launch attribute is safe (they do not touch their inputs before the previous kernel has
completed); without the attribute the instruction does nothing. They never call
`griddepcontrol.launch_dependents`, so a successor launched with the attribute starts only
when they finish.

## Measured state (scored run 36b5b559b0344d0f, 1.990x on the standalone problem)

| T | policy | this kernel (us) | previous kernel, same judge (a49578e5c30348bb) |
|---|---|---|---|
| 1 | sk8w4s | 29.3 | 46.7 |
| 8 | sk8w4s | 68.4 | 87.5 |
| 16 | sk8w4s | 97.3 | 107.4 |
| 32 | sk8w4s (split jobs) | 133.9 | 145.5 |
| 128 | legacy128 (the previous kernel's configs) | 255.1 | 250.4 |
| 256 | main-d2 (fat path) | 308.4 | 366.5 |

Every variant passes the FP32 self-test at M = 1 to 257 on normal and skewed routings, so the
kernel can be switched on at every T; at T = 128 the policy itself falls back to the previous
kernel, so the layer loses nothing there. In the layer, keep the bench in `prepare` (its
routing differs from the standalone problem's, and the choice between `legacy128`, `main-d2`
and `sk8w4s` at 32 to 256 tokens is within 10%), and trim the build: `MAIN_MODES = (0,)` and
`SKINNY_VARIANTS = ((8, 4, 0), (16, 4, 0))` cut the 140 s build to about half, which the
layer's own 300 s prepare budget needs.

## Where the numbers round to bf16

These follow SGLang's Marlin path, which the activation rule in TASK.md pins:

- GEMM1 accumulates in FP32 over the full K (inside one CTA on the skinny path, over the
  stream-K shares on the fat path). Gate and up are rounded to bf16 before SiLU
  (`bf16r(acc * gs13 * 128)`), SwiGLU's output is stored as bf16 (`inter` or `xb`).
- GEMM2 accumulates in FP32; its output is rounded to bf16 before the norm
  (`bf16r(acc * gs2 * 128)`), then multiplied by `(1 + w)` in FP32 and stored as FP32 in
  `ybuf`; the sum of squares uses those FP32 values.
- The final pass sums `topk_w * rsqrt(ss / 4096 + 1e-8) * y` in FP32 and rounds once to
  bf16.
- The dequant itself is exact: an E2M1 code times its E4M3 block scale is formed in bf16 with
  no rounding (at most 6 significant bits), and the global scale times 2^7 is applied in FP32
  in the epilogue.

## What the layer must not change

- Pair-row order is internal (expert-major, then entry order); outputs do not depend on it.
- Sums of squares are FP32 atomics, so bitwise results vary run to run at the last bit.
- The routing prologue assumes `topk_ids` holds global ids and filters to the local range;
  pass the ids the router emits, not local ids.
- The x tiles (xa, xb) cross from generic stores to other CTAs' `cp.async.bulk` reads. Each
  writer fences with `fence.proxy.async.global` before its release and each issuing thread
  again after its acquire; the gather publishes with `red.release` after its barrier. Keep
  all of it (docs/runs-moe.md, "Proxy ordering of the x tiles").
- The layer's copy is generated by `tools/dl_moe_inline.py`, which builds only the fat modes in
  its `LAYER_MODES` (modes 0, 16, 26 and 28 since layer commit 655b189; the layer benches them on
  judge-like routing, `_moe_judge_routing`). The layer's host wrapper zeroes `Params::cost`
  (the compiled stage-cost table); the struct is not zero-initialized, so a port must set it.

## PDL secondary in the layer (layer patch 2, 2026-10-29)

`tools/dl_moe_inline.py` applies a second layer-only patch to the inlined copy (the standalone
in `muse_moe_w4a16/` is unchanged). `Params` gains `int pdl`, which only the layer's host wrapper
sets (`moe_fused_run(..., pdl)`; only the first 512-token chunk takes it). With `pdl` set:

- the launch carries `cudaLaunchAttributeProgrammaticStreamSerialization` (`moe1_launch`);
  without it the launch is the plain `<<<>>>` it was;
- before `griddepcontrol.wait`, each CTA runs a preamble that reads nothing the primary writes:
  the ring's mbarrier initialisation (fat: `wfull`, `wempty`, `xfull`, `xempty`; skinny: `full`),
  with its `fence.mbarrier_init`, moved there from the routing section (which skips it when `pdl`
  is set), and `cp.async.bulk.prefetch.L2` hints of weights picked from the local expert index
  alone: unit 0 (16 stages of 36,864 bytes) of every local expert on the fat path, the first 4
  units on the skinny path, spread over the CTAs. A hint only; no result depends on it.

The layer uses it after its route kernel (`route_fwd`, front variants `concat-r` and `skf-r`),
which runs `griddepcontrol.launch_dependents` at entry.

Why correctness and progress never depend on the primary and the MoE being co-resident (PDL
overlap is opportunistic: the CUDA programming guide's programmatic dependent launch section
lets the secondary start early but does not promise it):

1. The secondary's CTAs can be scheduled only after every CTA of the primary has executed
   `griddepcontrol.launch_dependents` or exited, so all primary CTAs are already resident or
   done; a secondary CTA never keeps a primary CTA from an SM.
2. The route kernel waits on nothing the MoE does, so its CTAs always finish.
3. Before `griddepcontrol.wait` a MoE CTA touches only its own shared memory and issues
   fire-and-forget prefetches; it spins on no other CTA. Every cross-CTA wait (the x-row flags,
   `done_as`, the fixup slots, the grid barrier on `ctr[4 + gen & 1]`) and every read of the
   inputs (`topk_ids`, `topk_w`, `x`, `ctr[3]`) comes after the wait.
4. `griddepcontrol.wait` returns only when the primary grid has completed and its memory
   operations are visible (PTX ISA 9.7.15.14), so by then no route CTA holds an SM. The grid
   barrier needs all `sms` MoE CTAs resident at once, which holds as in the plain launch: one CTA
   per SM by the shared-memory footprint, and nothing else of ours is in flight (no side stream
   in the route variants; the next kernel is not launched early, since the MoE never triggers).
5. If the hardware does not overlap the launches, the MoE starts after the route kernel ends,
   exactly the plain launch's order.

SASS (`tools/dl_moe_sass_counts.sh`, CUDA 12.8, sm_90a) against the layer copy before the patch:
per kernel one more `FENCE.VIEW.ASYNC.S` (the preamble's `fence.mbarrier_init`), 16 more `SYNCS`
on the fat kernels and 4 on the skinny ones (the preamble's mbarrier inits), one `UBLKPF` (the
prefetch); `MEMBAR`, `BAR`, `ERRBAR`, `CCTL`, `UBLKCP`, `ACQBULK`, `LDL` and `STL` counts equal.
Registers and spills equal on all six kernels (main<28> 96, 80 B frame, 184 / 512 B).

Update (same day, layer commit after the mode 37 merge): `LAYER_MODES` is (0, 16, 26, 28, 37) and
the layer also builds the skinny kMode 3 (`sk8w4s-er`, `LAYER_SKINNY_EXTRA`). The skinny block now
initialises both `full` and the per-slot `empty` barriers in one block, so the skinny preamble
initialises both, and both preamble anchors now match the whole init block, so a barrier added
upstream fails the inliner instead of going uninitialised under `pdl`. SASS against the
standalone build of the same instances (`tools/moe_compile_fused.sh` with (8, 4, 3) added,
`tools/dl_moe_sass_counts.sh`): main<37> FENCE.VIEW.ASYNC.S 22 / 21, SYNCS 154 / 138, and
MEMBAR 43, BAR 15, ERRBAR 22, CCTL 8, UBLKCP 4, ACQBULK 1, LDL 104, STL 39 equal; skinny kernels
one more FENCE.VIEW.ASYNC.S and 8 more SYNCS (the preamble's 4 + 4 inits), BAR, MEMBAR, ERRBAR,
CCTL and UBLKCP equal. Registers: main<37> 96 with a 56 B frame and 156 / 416 B spills, as the
standalone; sk8w4s-er 161 against 159 standalone, no spills.

Check of the mbarrier inits under `pdl` (2026-10-29, after a review question): the layer MoE uses
`full` and `empty` in every skinny variant (sk8w4s, sk16w4s, sk8w4s-er) and `wfull`, `wempty`,
`xfull`, `xempty` in every fat mode (0, 16, 26, 28, 37); those are all the `mbar_init` targets in
`_MOE1_SRC`. The preamble initialises each with the routing block's arrival count (skinny `full`
1 and `empty` kNT, fat `wfull`/`xfull` 1 and `wempty`/`xempty` kRel), then
`fence.mbarrier_init`, on the thread that does it in the routing block (warp kNW lane 0, warp
kNWarpsM - 2 lane 0); the routing block is skipped only under `pdl`. That thread's own first use
(the skinny first-unit copies) follows in program order, every other thread's follows the
post-routing `__syncthreads`, as in the plain path. The a7e7594 inliner added `empty` to the skinny
preamble when the standalone block gained it; the earlier anchor matched only the `full` line, so
it now covers the whole block. `_moe1_pdl_self_test` (run in the route self-test) launches every
passing variant as the route kernel's PDL secondary, eagerly and as a graph replayed 100 times,
against the plain launch on the same route output; a failing variant the policy uses turns the
route variants off.

Update (mode 47, main 5e30c76): `LAYER_MODES` (0, 16, 26, 28, 37, 47), `47:
"main-lean-grp-nohoist"`; inliner output byte-identical on a second run. SASS against the
standalone build of the same instances: main<47> adds one FENCE.VIEW.ASYNC.S, 16 SYNCS and one
UBLKPF (the PDL preamble), every other ordering and memory opcode count equal (MEMBAR 43, BAR 15,
ERRBAR 22, CCTL 8, UBLKCP 4, ATOMG 148, REDG 9, LDL 22, STL 1); 96 registers, 8 B frame, 4 / 88 B
spills, as the standalone. The other nine kernels are unchanged from the sk8w3s merge.

Update (sk8w3s merge): the standalone skinny init block gained `hand` barriers (kHand, kMode low
bits >= 6, not built in the layer) and the fixed anchor stopped matching, as intended. The
inliner now copies the routing warp's init block itself into the skinny preamble
(`skinny_init_block`; every line of the block must be an mbarrier init, a loop, an
`if constexpr`, a brace or the fence, or the inliner fails), so the two paths initialise the same
barriers by construction. Layer variants: sk8w4s, sk16w4s, sk8w3s, sk8w4s-er and fat 0, 16, 26,
28, 37. SASS against the standalone build of the same instances: every kernel adds one
FENCE.VIEW.ASYNC.S, one UBLKPF and the preamble's inits in SYNCS (fat 16; skinny 8, sk8w3s 6);
MEMBAR, BAR, ERRBAR, CCTL, UBLKCP, ACQBULK, ATOMG, REDG, LDL and STL are equal; registers equal
(sk8w3s 130) except sk8w4s-er (161 against 159), no new spills.
