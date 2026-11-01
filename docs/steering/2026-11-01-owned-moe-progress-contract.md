# Owned layer MoE51 launch-progress audit — 2026-11-01

## Finding

The owned verify backend has a pre-existing unproved inter-CTA progress dependency. Current source and passing receipts do not establish a CUDA co-residency guarantee. This is separate from comment packing: selected device instruction equivalence passes, while complete production eligibility does not follow from it. No code or GPU action occurred.

Source pins: original backend399155B `6de124667ee5b56c81771b4fd5d2591dc62e1bcfbf60f18baa680438d543bf42`; final packed backend326021B `c1c6065a095efb407a15c8a1fe8f6936fc3078750b1ca7adb27447846342205e`. Root/owner frozen paths are `.build/today-proj/prod-size/frozen/owned-15-{original,comment}.py`. Compiler report's final-marker relation proves final device bytes unchanged; host launch policy was not fixed by packing.

## Exact dependency

Original source `_MOE1_SRC::moe1_launch` uses ordinary triple-chevron launch when `!p.pdl`; otherwise `cudaLaunchKernelEx` receives exactly one attribute, programmatic stream serialization. Neither branch requests cooperative launch. `_MOE1_HOST_SRC` explicitly describes grid=SMcount with no cooperative attribute (original line6962). `_owned_kernel_fn` calls `_moe1_run(...,10,True)` (line8631); first chunk sets `p.pdl=1`, while later chunks use zero. Dispatch resolves native main51 for the selected128/256 paths.

`moe_main_kernel<51>` has a software whole-grid completion barrier: tid0 fences, increments `bc`, waits for `bc>=G`, then the CTA joins and begins resetting shared global completion state (original lines6843–6867). Timeout breaks the wait and proceeds: bounded spinning avoids an infinite host hang but is not a proof of valid output/state when an unscheduled CTA has not arrived. Producer code also waits for per-job `xready` and `done_as` published by other CTAs (original6330,6355,6395). Such waits require a progress argument, not just local memory-order fences.

The source comment's claim that shared footprint keeps every CTA resident is an empirical scheduling assumption. grid=sms limits work size but is not a launch contract requiring ordinary blocks to reside concurrently. Joining a side stream and PDL grid-dependency wait can resolve predecessor data availability; neither grants all CTAs in this new grid co-residency. No source found in this bounded review converts the ordinary/PDL launch into cooperative launch. Existing standalone work explicitly added common cooperative admission in both comparator arms (`docs/runs-moe.md` owned-multiply entry835253441a214a1c); that does not repair the separate layer copy.

## Authoritative CUDA12.8 contract

NVIDIA states “Thread blocks are required to execute independently.” Ordinary blocks may be scheduled serially. [Programming Guide2.2](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#thread-hierarchy).

Its grid-synchronization section uses `cudaLaunchCooperativeKernel` and a launch-size/occupancy bound to guarantee co-residency; its SM-count example is a cooperative launch, not ordinary triple-chevron. [Programming Guide11.7](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#grid-synchronization-cg).

The12.8 launchEx API distinguishes the cooperative attribute from programmatic stream serialization: the first selects cooperative semantics; the latter permits resolving stream dependency programmatically and opportunistic overlap with a predecessor. Thus PDL is not a replacement for cooperative admission. [Driver API, cuLaunchKernelEx](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EXEC.html).

These docs establish a missing guarantee, not a claim that every measured H200 launch deadlocks. Old passing tests establish empirical success for covered executions only. Kernel fences/acquire-release publication remain independently necessary even after a proper launch repair.

## Minimal conservative repair gate, not implemented

For every main51 launch, query cooperative support and actual selected-function occupancy using its exact block size and dynamic shared allocation; reject unsupported/nonadmissible grids before launch. Use a checked cooperative launch on the actual current/capture stream. Safest first repair disables MoE PDL and uses ordinary same-stream predecessor completion; this avoids assuming combined cooperative/PDL support while preserving device math, all fences/barriers and ownership. A combined mode would need its own documented support/source proof and actual checked launch gate.

Do not silently fall through on a failed attribute/occupancy/launch check. Require private-state initialization, graph capture/replay and occupied-stream startup coverage, fresh math/reference checks and all selected native paths. Admission and capture compatibility must be checked on H200; local sm90a device compilation cannot validate host API behavior. Existing inter-CTA publication/reset protocol still needs its independent proof; cooperative admission repairs only progress/co-residency. No fence removal, weaker tolerance or numerical shortcut is proposed.

Stop public eligibility claims until the host-progress gap and any separately open dataflow/ownership defects are resolved. Source equivalence and byte fit can freeze independently without claiming this old policy was already safe.
