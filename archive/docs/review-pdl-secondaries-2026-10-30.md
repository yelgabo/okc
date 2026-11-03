# Review: every PDL secondary in muse_decode_layer

Read-only audit, 2026-10-30. It closes the caveat in `docs/review-m1-ordering-2026-10-30.md`,
which checked only the skinny projection kernel and attention. Nothing was compiled or run.

## Result

No issue on the shipped 2.014x release (main `muse_decode_layer/submission.py`, SHA256
`b9c23bda...`, the board entry's file). Every kernel the release launches with
`cudaLaunchAttributeProgrammaticStreamSerialization` executes `griddepcontrol.wait` in every
thread that writes global memory, before that write, and at least one thread of every CTA
that can exit waits. No pre-wait read has a writer that can still be running. The M1
build (6377b5a through `tools/dl_gate_patch.py --m1 --m1-runtime`) changes only `sk_kernel`
and attention's trigger, which the earlier review covered. The rest of its CUDA source is
identical to the release, so the verdicts below hold for both builds.

There is one latent ISSUE, in a kernel the release does not run on the scored path:
`w4_gemm_kernel` with KIND 2 returns before its wait when the GPU has no local routes (see
"W4 kernels"). It is harmless today and has a one-line fix.

The caveat's premise therefore holds: in every PDL chain of the layer, a kernel's completion
implies the completion of every kernel before it on the stream, and no secondary writes
global memory before its wait. This keeps the M1 review's rcount-reset and allocator-reuse
arguments intact.

## Method

- Release: main at fba35bb (`git rev-parse` at the time of reading), whose file hash matches
  the manifest's board entry. M1: `git show 6377b5a:muse_decode_layer/submission.py` patched
  with `tools/dl_gate_patch.py --m1 --m1-runtime` in a scratch directory. `diff` against the
  release shows CUDA changes only in the attention kernel (`ONE_TRIG` trigger at entry) and in
  `sk_kernel` / `launch_sk` (`pdl_run`, the proxy fences, `M | (1 << 30)`). Everything else in
  the diff is Python (the gate, the M1 tuner, the `pdl` argument of the sk runners).
- Every `cudaLaunchAttributeProgrammaticStreamSerialization` site was found with `grep`, and
  each launch helper was traced to its callers and to the kernel_fn path that reaches it.
- The MoE kernels (`_MOE1_SRC`) and the W4 kernels were read by a separate agent. I re-read
  the wait, trigger and early-exit lines it cited (6984-6996, 7066-7075, 7228-7236,
  7353-7362, 8444-8452, 12534-12542) and confirmed that `_MOE1_SRC` contains no
  `launch_dependents`.
- Line numbers are the release file's. In the M1 file they shift by +7 (before line 3874), +16
  (3874 to 4523) and +21 (after 4523).

Two reading rules used throughout:

1. A thread that never executes the wait may still write global memory safely if it is
   ordered after a thread that did by a `__syncthreads` or cluster barrier. Threads that exit
   without waiting and without writing are harmless, provided another thread of the grid
   waits, because the grid then cannot complete before its primary does.
2. A trigger at entry is safe exactly when the secondary's pre-wait reads have writers that are
   finished. So question 3 (where the trigger sits) is answered by question 2 (the
   secondary's pre-wait reads).

## Which launches carry the attribute

| kernel | launch | attribute | on the scored release path |
|---|---|---|---|
| `route_align_kernel` (327) | `launch_ex(..., kPdlGlue)`, 1224 | yes (`kPdlGlue = true`, 320) | no: the fused MoE meta returns before `route_align` at every scored size |
| `finalize_row_kernel` (434) | `launch_ex(..., kPdlGlue)`, 1241 | yes | no (Marlin / W4 fallback only) |
| `router_kernel<M>` (537) | `router_launch`, 629 | yes | yes, T = 1 (front `fp32`, side stream) |
| `pre_norm_kernel` (648) | `launch_ex_pdl`, 1290 / 1294 | always | yes, T = 1 (`route_side`, main stream) |
| `route_kernel` (733) | `launch_ex_pdl`, 1341-1354 | always | yes, T = 8 to 256 (fronts `skf-r`, `concat-r`) |
| `proj_kernel` (1003) | `launch_ex_pdl`, 1175 | always | fallback only: `_proj` uses it when no sk/gemv pick exists for m <= 8; `router_mma` is not a scored front |
| `prep_kernel`, `gate_kernel` (1442, 1482) | `launch_ex_pdl`, 1515 / 1532 | always | no (`_attn_prep` path; `attn_one` wins every scored shape) |
| `attn_kernel` (fused attention, 1788) | `launch`, 2504 | always | no |
| `combine_kernel` (2468) | 2564 | always | no |
| `attn_ws_kernel` (2885), `ws_combine_kernel` (3605) | 3663 / 3717 | always | no |
| `attn_boundary_kernel` (6469) | `launch_ex(..., kPdlGlue)`, 6773 | yes | yes, T = 1 (c8x256) |
| `attn_boundary_geo_kernel<CL, NT>` (6575) | `launch_cl`, 6678 | yes | yes, T = 8 (c1x1024) and 256 (c1x512) |
| `boundary_kernel` (FFN boundary, 6394) | `launch_ex(..., kPdlGlue)`, 6811 | yes | yes, T = 1 and 8 (`bnd_ours`, T <= `SLICED_UNPOOL_MAX_TOKENS`) |
| `w4_gemm_kernel` (6959), `w4_swiglu_kernel` (7353), `w4_finalize_kernel` (7229) | `launch_pdl` 7398, `finalize_launch` 7451 | when `_w4_policy` returns PDL | no |
| `moe_fused_kernel`, 4 copies (8414, 9115, 9816, 10517) | `moe1_launch`, 7607-7626 | only when `p.pdl` (first 512-token chunk, `meta["pdl"]` set by `_moe_front_route`, 17872) | yes: sk8w3s is the 9816 copy, PDL at T = 8 and 16, plain at T = 1 |
| `moe_main_kernel<47>` (12510) | `moe1_launch` | as above | yes, T = 32, 128, 256 (PDL) |
| `attn_one_kernel` (4478) | `one_launch`, 6160 | only if `_ONE_PDL` | no: `_ONE_PDL = False` in both builds (16514; M1 16847), so attention is a plain launch |
| `sk_kernel` (13403) | `launch_sk`, 14036 | only under `SK_PDL` with bit 16 | M1 only; covered by the earlier review |

Launched plain, so not secondaries: the cgemv / rgemv / sgemv kernels (14299-14452), `wd_kernel`
(14849, cluster attribute only), `prefetch_kernel` (1379; `prefetch_ours` is False by default),
`sk_kernel` in the release, and every torch, cuBLAS and SGLang kernel.

Because attention is a plain launch in both builds, and the release's sk kernels are plain as
well, the PDL chains in one layer call are short segments, each of which starts after a full
stream-order barrier.

## Per kernel

### Glue kernels in `_CUDA_SRC`

**`route_align_kernel` (327-410): SOUND.** It triggers at 333 and waits at 339, with no
global access between them, unconditionally in every thread and with no earlier return. It
reads `start_p` (`local_start`, a harness input) and `topk_ids` after the wait. It writes
nothing before the wait. There is no co-residency dependence (one CTA).

**`finalize_row_kernel` (434-516): SOUND.** It triggers at 438 and waits at 439, at entry.
No exits.

**`router_kernel<M>` (537-627): SOUND.** It triggers at 540. Before the wait (559) it reads
only the FP32 router weights (`__ldcs`, 553-556), which are prepare-time state. `x` (the
attention boundary's router copy) is read after the wait. All threads wait and there is no
early exit. The output is written after the wait. The two `cluster.sync()` calls are within
one cluster, and the hardware co-schedules a cluster. On the side stream the kernel's
captured dependency is the event after the attention boundary, not the previous side-stream
kernel. Whether capture makes that edge programmatic does not matter, because nothing the
boundary writes is read before the wait.

**`pre_norm_kernel` (648-725): SOUND.** It triggers at 651. Before the wait (668) it reads
only `w` (`pre_expert_w`, prepare). `x` is read and `out` is written after the wait. There are
no exits; `pre_norm` skips the launch when m == 0 (1282).

**`route_kernel<InT, HILO>` (733-888): SOUND.** It triggers at 739. Before the wait (763) it
reads `w` (prepare) and, in warp 0, `bias` (`route_bias`, prepare). Logits and the pooled rows
(the front GEMM's or skf's output) are read after the wait. Warps 1-3 return at 838, after the
wait and after their `out` stores. `route_fwd` skips the launch when m == 0. This matches the
provenance table in `docs/runs-decode-layer.md` (around line 330).

**`proj_kernel` (1003-1160): SOUND.** It triggers at 1006. Before the wait (1070) it issues
`cp.async` of packed weight tiles only (`load_w`, 1051-1056), which come from prepare. The x
rows of the prefetched stages are read with plain loads after the wait (1071-1083). There are
no early returns. All stores to `out` happen after the cluster reduction, after the wait.

### Attention prep and the non-`one` attention paths (not on the scored path)

**`prep_kernel` (1442-1478): SOUND.** It triggers at 1447. Warps with `h >= heads` return at
1453 before the wait, and they write nothing. Every CTA has at least one waiting warp: with 20
heads in 8-warp CTAs, the third CTA's warps 16-19 wait. No global reads happen before the
wait. The cache rows at `loc[t]` are written after it.

**`gate_kernel` (1482-1502): SOUND.** It triggers at 1485. Warps with `h >= HQ` return before
the wait and write nothing; with HQ = 16 and 2 x 8 warps, none actually do. The wait is at
1493.

**`attn_kernel` (fused attention, 1788-2466): SOUND.** It triggers at 1790. Before the wait
(1881) it reads `k_scale`, `v_scale`, `seq_lens`, `page_table` and cached K/V rows below
`seq_lens`. These are harness inputs written before the step, and the new token's rows are
written by this kernel after the wait (2006-2008). The `return`s around 2236-2260 are inside
lambdas, not kernel exits. All threads pass the wait.

**`combine_kernel` (2468-2480), `ws_combine_kernel` (3605-3620): SOUND.** Both trigger and
then wait at entry, with no exits.

**`attn_ws_kernel` (2885-3600): SOUND.** It triggers at 2886. The loader warps (`warp >= 8`,
2940-3050) never execute the wait. They read only the page table and the cache rows of earlier
tokens (harness inputs), write only shared memory (plus `p.dbg` under the compile-time `STAMP`
debug flag), and return at 3050. The consumer warps of the same CTA wait at 3064 before reading
`qkv` and before writing the cache, the workspace or the output, so the grid cannot complete
before its primary does.

### Boundaries in `_BND_SRC`

**`attn_boundary_kernel` (6469-6560): SOUND.** It triggers at 6475. Before the wait (6488) it
reads `stream` (the step's input `data["stream"]`), `alpha`, `beta` and `w_next` (prepare).
The branch (the o projection's output) is read after the wait. No exits. The stores come after
the second `cluster.sync()`. The stream input is not written inside the step. If `.contiguous()`
had to copy it, the copy would be a plain torch kernel right before this launch, and that
kernel never triggers early, so the copy would be complete before this kernel starts.

**`attn_boundary_geo_kernel<CL, NT>` (6575-6660): SOUND.** It triggers at 6581. Before the
wait (6599) it reads only `stream`, as above. `alpha`, `beta` and `w_next` are read after the
wait. No exits. Cluster syncs are compiled in only when CL > 1.

**`boundary_kernel` (FFN boundary, 6394-6460): SOUND, with a chain argument.** It triggers at
6400. Before the wait (6419) it reads `alpha`, `beta`, `w_next` and `w_post` (prepare), and
`stream`, which is the attention boundary's `stream_out`. That one is a predecessor's output
and not prepare-time data. The release-path primary is the unpooler. When `_proj` has an sk or
gemv pick for it (the tuner's normal case at 1 and 8 tokens), the unpooler is a plain launch,
so everything before it, the attention boundary included, is complete before the FFN boundary
can start. When it falls back to `proj_kernel` (PDL, trigger at entry), completion of the
attention boundary still comes first, along every front:

- `fp32` (T = 1): the main stream joins the side stream (`main_stream.wait_stream`) before
  the MoE. The side stream's router waits on the attention boundary, and its later kernels
  are plain. The MoE node therefore has a full edge to a node that completes only after the
  attention boundary has completed.
- `skf-r`: in the release the skf sk kernel is plain, so it starts after the attention
  boundary completes. In M1 it is a PDL secondary that waits in every thread and never
  triggers early, so `route_kernel` starts only after skf's CTAs have exited, which happens
  after their wait has seen the attention boundary complete.
- `concat-r` (only at 128 / 256 in the release, where the FFN boundary is SGLang's): the
  cuBLAS GEMM is followed by `route_kernel` and the fused MoE. The MoE waits in every thread
  before anything else and never triggers early, so the unpooler cannot start before the MoE
  has completed, which implies `route_kernel` and the GEMM have completed.

The source comment at 6406-6409 attributes this to "GEMM1 or Marlin, which wait before they
trigger". For the fused MoE the precise reason is that it waits and never triggers early.
Either way the pre-wait read of `stream` is of finished data. No exits; stores come after the
last `cluster.sync()`.

### MoE (`_MOE1_SRC`)

**`moe_fused_kernel`, all four copies (8414 / 9115 / 9816 / 10517; 9816 is sk8w3s): SOUND.**

- **Wait.** It is unconditional in every thread (8450 / 9151 / 9852 / 10553). Before it there
  is only the `if (p.pdl)` preamble, with no return or branch out. The kernel has no `return`
  at all. The timeout (`spin_failed`) paths are after the wait and only break loops, and every
  CTA ends at the grid barrier.
- **Before the wait.** It does mbarrier init and `fence.mbarrier_init` (shared memory) and
  issues `cp.async.bulk.prefetch.L2` of `w13` units (prepare). It makes no global load into
  registers or shared memory, and no global write or atomic.
- **After the wait.** `ctr[3]` (the generation), `topk_ids`, `local_start` (`ld.global.nc`),
  `x` and `topk_w` are read. `ss_next` and `ctr[4 + next]` are reset.
- **Trigger.** None in `_MOE1_SRC`, so the trigger is implicit at exit.
- **Co-residency.** The grid barrier and the `done_a` spins need all G (SM-count) CTAs
  resident. Under PDL this holds: the MoE launches only after every `route_kernel` CTA has
  triggered (739). `route_kernel` never waits on the MoE, so its SMs drain, and no successor
  can take SMs early because the MoE never triggers. The kernel spins on no flag written by
  another grid.

**`moe_main_kernel<47>` (12510; T = 32, 128, 256): SOUND.** It has the same structure. The
wait at 12541 is unconditional, with only the `p.pdl` preamble before it (mbarrier init, and
an L2 prefetch hint of `w13f` units from prepare). There is no `return`. Every warp role
(weight producer, x producer, consumers) begins after the wait and a `__syncthreads`. The
xready / done_as spins and the grid barrier are after the wait and within one grid. There is
no trigger.

Because neither MoE kernel triggers early, the provenance of `ctr[3]` and `ss` from the
previous MoE launch has a stronger basis than the runs-doc table gives: no later kernel can
start until the MoE has exited.

### W4 kernels (not on the scored path)

The fused MoE is picked at all six scored token counts, so `_experts_run` returns through
`_moe1_run` (15506). The W4 kernels run in prepare benches and as the fallback when the fused
MoE is unavailable.

- **`w4_gemm_kernel`, KIND 1 (GEMM1): SOUND.** It waits at 6988 and triggers at 6989, before
  `num_items` is read and before the `q >= total` return (6993-6995).
- **`w4_gemm_kernel`, KIND 2 (GEMM2): ISSUE (latent, not on the scored path).** It reads
  `num_items[0]` (6991), `items` and weights and scales before its wait (7072). Those reads
  are fine: `route_align` finished before GEMM1 passed its own wait, and GEMM1 triggers only
  after that wait. But `if (q >= total) return;` (6993-6995) runs before the wait. When this
  GPU has no local routes (`num_items2 == 0`), every GEMM2 CTA exits without waiting, so the
  grid can complete before `w4_swiglu_kernel` and GEMM1 complete. That breaks the transitivity
  that `w4_finalize_kernel` and everything after it rely on. With top-8 of 256 experts and 32
  local experts, a 1-token call has no local route with probability (224/256 choose 8)
  / (256 choose 8), about 0.34. Today nothing goes wrong. On that path GEMM1 and swiglu write
  nothing (no items, every `local_ids` negative), and their only live reads are of buffers
  that stay allocated. A later change that writes in those kernels unconditionally, or an
  allocator reuse of a buffer they read, would race. Minimal fix:

  ```cpp
  if (q >= total) {
      if (KIND == 2) {
          asm volatile("griddepcontrol.wait;" ::: "memory");
      }
      return;
  }
  ```

- **`w4_swiglu_kernel` (7353): SOUND.** It triggers at 7358 and waits at 7359, before the
  `local_ids[p] < 0` return.
- **`w4_finalize_kernel<CL>` (7229): SOUND.** It triggers at 7233 and waits at 7234, at entry.

## Answers to the caveat

- **Every secondary waits before its first global write and before exit.** This holds for
  every PDL-launched kernel on the release and M1 paths. Some threads exit without waiting:
  the `prep_kernel` and `gate_kernel` surplus warps and the `attn_ws_kernel` loader warps. They
  write nothing to global memory, and another thread of the same CTA waits. The only CTA-wide
  exit before the wait is in W4 GEMM2, above.
- **No pre-wait read of a predecessor's output.** Every pre-wait read is of prepare-time state,
  harness inputs or launch parameters, with one exception: the FFN boundary's read of the
  attention boundary's `stream_out`. That writer is complete by the chain argument above.
- **Triggers.** The glue, boundary, projection, prep and attention-path kernels trigger at
  entry. The MoE kernels and `sk_kernel` never trigger. W4 GEMM1 and GEMM2 trigger after their
  wait. Every secondary of an entry-triggering kernel has safe pre-wait reads, so no trigger
  placement is a hazard.
- **Co-residency.** No kernel spins on a flag written by another grid. The intra-grid
  barriers in the MoE kernels and attention, and the cluster syncs, need only their own grid's
  CTAs, which the arguments above show become resident.

## Not checked

- Builds other than the two named. A file built with `ONE_TRIG` or `_ONE_PDL = True` would
  make attention a secondary, and its waits were checked only in the earlier review.
- SGLang, cuBLAS and Marlin kernels. Whether any of them launches with PDL internally under
  capture was not examined. The arguments above do not depend on it, because each of our
  secondaries waits before reading their output and our chains are ordered by our own kernels'
  waits.
- Whether stream capture turns the cross-stream event edge into the router into a
  programmatic edge. The router is safe either way.
- Stale-L1 questions for `__ldg` / `ld.global.nc` reads issued after a wait. No such line is
  cached before the wait in any kernel above.
- Ordering inside a kernel (mbarrier phases, proxy fences inside the MoE kernels).
- No compile, SASS read or GPU run.
