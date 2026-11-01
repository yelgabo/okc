# Addendum: owned51 launch progress is a separate eligibility blocker

**The comment-packed hybrid is not ready for a new measurement or public release on the current owned launch contract.** The source/assembly equivalence clearance in [the host review](2026-11-01-hybrid-packed-host-review.md) remains valid; equivalence preserves this pre-existing launch-contract gap too. No owned kernel/protocol/launch was edited during this audit.

Exact reviewed owned-both backend: `c1c6065a095efb407a15c8a1fe8f6936fc3078750b1ca7adb27447846342205e`, embedded in mask15 `d3f2b31e…` and the corresponding owned32 body in mask4 `3b205d40…`. Root independently identified this issue after source packing passed; this addendum confirms it against the real source and the earlier repository proof.

## Executed path and missing guarantee

Actual owned `kernel_fn` calls `_moe1_run(...,10,True)`. The real `_moe1_source()` installs case10 `launch_main<51>`. `_moe1_run` forwards the `pdl` argument to `moe_fused_run`; its first chunk sets `p.pdl=1`. `_MOE1_SRC::moe1_launch` uses `cudaLaunchKernelEx` with exactly one attribute, `cudaLaunchAttributeProgrammaticStreamSerialization`. Its PDL-false branch uses ordinary `kern<<<grid,block,smem,stream>>>`. Neither branch requests cooperative launch/admission. `launch_main` sets maximum dynamic shared memory but does not query occupancy or cooperative support. The host grid uses device SM count; that is a chosen size, not a launch-level co-residency guarantee.

The selected MAIN51 body needs other CTAs: it polls activation-readiness/stage counts (`xready`/`done_as`) and, after its CTA joins, increments then waits for all G arrivals on `ctr[4+(gen&1)]` before cleanup. Large shared memory can limit the number resident per SM; it does not by itself require the runtime to admit every grid CTA simultaneously or prevent competing launch occupancy. The bounded `spin_failed` watchdog sets `ctr[8]` and permits fall-through; it does not make missing inter-CTA data or early cleanup correct.

## Primary contract and earlier evidence

CUDA12.8 §2.2 permits thread blocks to run in arbitrary order, including serially. §8.7 uses cooperative launch and an explicit resident-grid size calculation for grid-wide synchronization/co-residency. §3.2.8.6.2 describes PDL overlap as opportunistic and warns against relying on concurrency for progress. These guarantees do not turn an ordinary sms-sized launch into a cooperatively admitted grid. This is a source-contract conclusion, **not a newly observed H200 deadlock**. [NVIDIA CUDA12.8 Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#grid-synchronization-cg).

The repository already documented this exact distinction in [owned fallback admission](2026-11-01-owned-moe-fork-fallback.md), section “Optional owned cooperative launch”: ordinary launch does not establish required co-residency. `tools/dl_fork_owned_cooperative.py` provides a separately reviewed optional host helper requiring SM90 cooperative support, actual kernel/block/shared occupancy, bounded grid/chunk/local shape, fatal CUDA errors and `cudaLaunchCooperativeKernel`, rejecting PDL. That optional helper was not part of this historical owned verify backend. The later [owned multiply comparison](2026-11-01-owned-multiply-decode.md) likewise deliberately used a common cooperative/PDL-off control rather than treating old passed tests as a proof.

## Required next step, not performed here

Root must separately repair/qualify the reachable owned launch host path with cooperative admission and PDL off (or produce an equally strong primary-contract alternative). Retain every device fence/barrier/math boundary, scratch/Params ABI and chunk policy. Require exact-variant occupancy/admission checks, fatal errors, graph compatibility and replay correctness. Both sides of any matched comparison must use the same repair; performance against historical ordinary/PDL bytes is then a changed launch policy and cannot be called byte-identical. Regenerate new hybrid and affected private comparison pins explicitly. An unchanged packed source or an old H200 pass does not satisfy this gate.

Until then preserve the queued diagnostics' intended order but do not silently mark the owned four-cell/production hybrid eligible. Parent owns whether to revise a payload or its queue entry. No additional GPU run, automatic retry, native edit or public action is authorized by this addendum.

Root citation-label correction, November1: the linked CUDA12.8 grid-synchronization section is11.7; the earlier8.7 label reflects an older guide layout. The missing-admission finding is unchanged.
