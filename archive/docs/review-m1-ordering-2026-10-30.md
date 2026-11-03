# Review: M1 PDL ordering in muse_decode_layer

Independent read-only review, 2026-10-30, of the argument in
`docs/steering/2026-10-30-m1-ptx-ordering.md`. Nothing was run on the judge.

What I read:

- `git show 6377b5a:muse_decode_layer/submission.py`: `sk_kernel` (the `#if SK_PDL` block, the
  producer warp, escape warps, the fused router epilogue), `launch_sk`, `sk_fwd`, `skf_fwd`,
  `_skp_runner`, `_proj`, the `skf` call in the MoE front, `_m1_mode`, and the attention kernel's
  `ONE_TRIG` trigger, its `griddepcontrol.wait` sites and its early exits.
- `tools/dl_gate_patch.py` (FENCE, PRODUCER and RUNTIME_REPS patches) and
  `tools/dl_m1_bench_build.py`. I ran `dl_gate_patch.py 6377b5a.py OUT --m1 --m1-runtime` in
  the scratchpad. It applies cleanly, and the kernel diff consists of the two
  `fence.proxy.async.global` lines, the `Mf` / `pdl_run` changes and `M | (1 << 30)` on the PDL
  launch only.
- PTX ISA 9.4 (docs.nvidia.com/cuda/parallel-thread-execution, fetched 2026-10-29). Section
  numbers below are from that version: 8.6 Proxies, 8.9.1.1 Asynchronous Operations, 8.9.4
  Memory synchronization, 8.9.5 Causality Order, 9.7.10.28.2 Async Proxy, 9.7.15.4
  membar/fence, 9.7.15.14 griddepcontrol.
- CUDA Programming Guide, "Programmatic Dependent Launch and Synchronization".
- CUTLASS `include/cutlass/arch/grid_dependency_control.h` and
  `gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` (main, fetched 2026-10-29).

## Verdict: SOUND WITH CAVEAT

The issue order is correct for questions 1, 3 and 4 under the documented PDL semantics. On
question 2, the code is as safe as the documentation allows, but no fence placement can be
derived from the formal memory model, because `griddepcontrol.wait` has no defined place in it.
Keep the reader-side fences. No code change is required.

## 1. Pre-wait reads

Before `griddepcontrol.wait`, only thread 0 touches global memory. It runs
`prefetch.tensormap` on the `__grid_constant__` x map, a parameter fixed at launch. It issues
`cp.async.bulk` copies of `ws` (the packed weights from `_skp_pack` in prepare), `rws` (the
fused router pieces, also from prepare) and `ptab` (the escape table, also from prepare). The
attention kernel and the boundary kernel write none of these. The weight and table tensors
are written before capture and the host synchronizes before any timed replay, so 8.9.4 "API
synchronization" (stream task completion, then `cudaStreamSynchronize`) orders those writes
before every later kernel. That rule covers async-proxy readers too, because it says the API
mechanisms "participate in proxy-preserved base causality order". The pre-wait copies are the
same reads every sk launch in the release already makes. No thread stores to global memory
before the wait. The only pre-wait writes are to the CTA's own shared memory (mbarrier init,
`expect_tx`, the zero row).

In the PDL path the stage barrier accounting is correct. `issue_w` arms the full barrier with
`stage_tx = UPS*XU + WBY` (or `UPS*XU + RWBY` for a router stage) and posts only the weight
bytes. `UPS*XU = UPS*M*128` equals the TMA box `{64, M, UPS}` in bf16, so the phase cannot
complete until the post-wait x copy lands (9.7.15.16.6.2 complete-tx). No consumer, escape
warp or producer reaches an mbarrier wait before the CTA-wide `__syncthreads` and the wait,
so the `WAIT_LIMIT` self-timeout cannot be tripped by a slow primary.

Under `--m1-runtime`, bit 30 of `M` is set only in the `sk_pdl_launch` branch of `launch_sk`,
which is also the only launch that carries `cudaLaunchAttributeProgrammaticStreamSerialization`.
The patch count check (exactly one match of the 12-space `cudaLaunchKernelEx` line) leaves the
cluster branch untouched, and cluster configs have `PDL = false` anyway. Every launch that can
start early therefore executes the wait in every thread. A plain launch skips the wait, which
is correct because stream order already serializes it. The opposite mismatch, a PDL-attribute
launch without the wait, would be a race, and the code cannot produce it.

## 2. Visibility of the primary's generic stores to post-wait TMA reads

The 9.7.15.14 text says `.wait` "causes the executing thread to wait until all prerequisite
grids in flight have completed and all the memory operations from the prerequisite grids are
performed and made visible to the current grid." The Programming Guide says the same in API
terms: `cudaGridDependencySynchronize` blocks "until all primary kernels ... have completed and
flushed results to global memory."

The formal gap is that 8.9.4 lists the CUDA synchronizations that create a synchronizes
relation and that "participate in proxy-preserved base causality order": stream task
boundaries, graph node edges, events. `griddepcontrol.wait` is not in that list. The model
therefore gives no formal synchronizes edge from the primary's stores to the secondary's
reads, in any proxy. The author's reading 2 assumes a generic-proxy edge exists and then adds
the proxy fence. The model text supports neither reading 1 nor reading 2 by derivation. Both
rest on the instruction description.

On whether the fence is sufficient: with the fence, the path is the primary's generic `st`, then
the wait (performed and visible to the grid), then `fence.proxy.async.global` in the reading
thread's program order, then the `cp.async.bulk.tensor` read. 9.7.15.4 says bi-directional proxy
fences "take effect within a single thread" and their "cross-proxy synchronizing effect
composes with other forms of synchronization". 9.7.10.28.2 asks for a `fence.proxy.async`
between generic and async accesses to the same location and does not say on which side. So the
reader-side placement puts the required proxy fence on the path, which is as much as the text
can support. Every thread that issues x copies has one: thread 0 for stages `0..npre-1` and
the producer lane for stages `npre` onward. The escape warps and consumers read x only from
shared memory after the full barrier completes, and 9.7.10.28.2 says completion "is followed by
an implicit generic-async proxy fence", so they need no fence.

On whether the fence is necessary: probably not. 8.9.4 exempts the plain kernel-boundary case
explicitly, and every sk launch in the release reads its predecessor's generic stores through
TMA with no fence. NVIDIA's CUTLASS does the same thing after PDL: the sm90 warp-specialized
cooperative GEMM's mainloop producer calls `cutlass::arch::wait_on_dependent_grids()`
(`griddepcontrol.wait`) and then issues TMA loads of A and B with no proxy fence, under the
comment "Ensure that the prefetched kernel does not touch unflushed global memory prior to
this instruction". Mechanically, the primary has completed and its stores are performed at
GPU scope, the bulk copy reads through L2, and the reading thread has no generic-proxy access
to x that could hold a stale L1 line. This is precedent and mechanism, not a proof. The fence
costs two `FENCE.G` per CTA, per the author's SASS counts, so keeping it is the right call.

A writer-side `fence.proxy.async.global` in attention, after its output stores and before
exit, would match the repository's documented pattern (CLAUDE.md, evidence rules) more
closely. It has the same formal status, because the wait is still not in the model. I do not
require it: the release never had it for the plain-launch TMA read of the same buffer, and that
case is covered formally by 8.9.4.

## 3. Buffer reuse and WAR hazards

- Shared memory: each CTA has its own allocation. A co-resident secondary CTA cannot alias the
  primary's shared memory.
- Global writes by the secondary (`out`, `rpart`, `rcount`, `rout`, `diag`) all come after the
  wait in every thread. Every buffer the primary, or any earlier kernel, still reads is
  finished by then. This covers caching-allocator reuse: `out` is `torch.empty` per call and
  may be a block an earlier kernel used. The write happens after the primary completes, and
  the primary's completion implies its own predecessor's completion, because attention
  executes `griddepcontrol.wait` in every CTA that does work. The empty-range CTAs that exit
  early (`!SPEC && lo >= end`) write nothing.
- Kernels after the secondary: `sk_kernel` never executes `launch_dependents`, so the trigger is
  implicit at exit (Programming Guide: it "implicitly occurs after all thread blocks in the
  primary kernel exit"). Nothing later on the stream can start before the secondary's TMA reads
  of x finish. Consumers wait on every stage's full barrier before exiting. The exception is the
  timeout (`dead`) path, which already produces a diagnostic failure. A later attention that
  rewrites the x buffer, in the next layer call or a later replay, therefore cannot race these
  reads. The `side_stream.wait_stream` after `skf` records an event after the secondary, so it
  waits for the secondary's completion as well.
- `rcount` is self-resetting across launches (the last arrival zeroes it). The next `skf`
  touches it only after its own wait. The previous `skf` is complete by then provided every
  PDL secondary between the two executes the wait on its completing path. I checked that for
  attention only. The other secondaries (pre-norm, boundaries, router, route_align, W4 GEMMs)
  are the existing "not re-audited" item in the author's section 5. That is a property of the
  release's blanket PDL chain, and M1 does not introduce it.
- CUDA graph replay: programmatic edges exist only between adjacent kernel nodes inside one
  graph. Successive replays on one stream are full task boundaries (8.9.4), so no overlap
  crosses a replay.

## 4. Forward progress

The secondary launches only when every primary CTA has executed `launch_dependents` or exited
(9.7.15.14). With `ONE_TRIG` at kernel entry, every attention CTA is resident before any sk
CTA can occupy an SM. That covers the cooperative mode's software grid barrier, which needs all
splits resident. Secondary CTAs block in hardware at the wait, and the primary never waits on
them. Secondary CTAs that do not fit stay queued, and nothing depends on them. If the
attention instance is not a `ONE_TRIG` instance, or something else sits between attention and
o (a `.contiguous()` copy, for example), the trigger falls back to completion, which is slower
and still correct. Correctness never depends on co-residency, in line with the guide's warning
that relying on concurrent execution "is unsafe and can lead to deadlock".

## Caveats

1. The formal model (PTX ISA 9.4, 8.9.4) does not cover `griddepcontrol.wait`. Question 2 is
   settled by the instruction text, the Programming Guide, and NVIDIA's own use in CUTLASS, not
   by a derivation. The reader-side fences remove the only plausible proxy objection at
   negligible cost. Keep them in every build that compiles the PDL path.
2. The `rcount` and allocator-reuse arguments in section 3 rely on every PDL secondary in the
   layer executing `griddepcontrol.wait` before it completes. I verified this for the sk
   kernel and for attention, not for the other secondaries. A path in one of them that exits
   without the wait would break transitive completion for the whole chain, independently of M1.
3. Passing the fresh-input gate is consistent with this analysis but does not prove it, as the
   author also states.

## Minimal fix

None required. If a reviewer insists on the repository's writer-side pattern, add
`fence.proxy.async.global` in attention after the final stores of its output (in
`finalize_rows` writers, before exit) for the `ONE_TRIG` instances. That is optional and has
the same formal standing as the reader-side fence.
