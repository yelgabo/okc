# Native MoE empty routing: separate repair and qualification handoff

2026-11-01. Evidence parent: `710f2b8` safety patcher; this report accompanies
the independent empty-routing commit. No GPU job, scored submission, native CUDA
compilation, or speedup measurement was performed for this repair. Measured G1b2
bytes are unchanged. The coordinator owns integration, index publication and H200
qualification. Native release remains blocked until the new conformance passes.

## Exact scope and proof

Input G1b2 file SHA256:
`258c848c662149f9c6ac178039f179de6f0b0ec4cb0b503451d2974c7a7a8dfc`.
Its AST `MOE_CUDA_SRC` literal is 155986 bytes, SHA256
`1540fe13d00083d90f9e51f84b6002d18c51c41a4d8259197cb7d62d73f76215`.

`tools/dl_fork_moe_empty.py::patch_empty_routing(source)->str` requires that
exact literal and exactly one assignment. It adds only
`if (nj == 0) return 0;` at the start of `wgs_stage_at`, before the prefix
pointer and job metadata access. After replacement it normalizes this one literal
and checks equality of every other AST node. Altered source and double application
fail closed. This also guards the direct `wgs_range` API.

Without the guard, the zero-job path reaches `S.jC[0]`, which the job builder did
not initialize. This is an invalid metadata read; it is not evidence that every
empty call necessarily miscomputes or hangs. `wgs_ranges` already writes zero
prefix endpoints/totals, performs its existing warp synchronization, and publishes
the four ranges. All four are now zero. Neither producer emits stages. Consumer
`wgs_phase` breaks before its job lookup and the sole `wgs_cta_of` call, so that
helper's division by total cost is unreachable on this path. No extra write to
`S.jrow` or inter-warp shared-memory ownership was introduced.

All routing publication joins, proxy fences, empty/full slot barriers, cooperative
grid rendezvous, generation resets and final output writers remain unchanged.
The finalizers mask every nonlocal slot via `S.pos==-1`, accumulate from zero, and
store BF16 zero. The guard changes no nonempty arithmetic or BF16/SC6 boundary.

Generated pure file: `.build/today-moe/g1b2_empty.py`, 998317 bytes, SHA256
`6c23c8f38d227ed50b3727e68d4b6b65ac3a0fcb0ed79be1fec43f8bef9064ab`.
It is an offline artifact, not a qualified submission candidate.

## CPU evidence and reproduction

Three source/C++ tests pass. An actual extracted original and patched C++ helper
were built with Zig 0.15.2 C++17 -O1 and executed on CPU: 181169 checks, including
both phases, every grid size 1..136 and every CTA endpoint in the empty case,
plus differential nonempty prefixes/jobs. Patched empty metadata accesses: zero.
The original helper's instrumented `jC[0]` access is observed by the negative
control. Nonempty helper outputs compare exactly. Cost functions are CPU stubs;
this proves guard reachability/nonempty identity, not the real cost model or GPU
execution. Root independently executed an earlier equivalent extracted mock.

`tools/test_dl_fork_moe_empty.py` defaults to tracked
`muse_decode_layer/cand/g1b2_native_experts.py`; set `OKC_EMPTY_BASE` to the
coordinator's exact G1b2 file in an unintegrated worktree. `OKC_EMPTY_ZIG` can
override `/root/.cache/okc-host-tools/zig-0-15-2/ziglang/zig`.
The test invokes WSL and puts its generated mock/binary in `.build/today-moe`.

Four actual Torch 2.10 CPU tests execute the real conformance function and its
summary/gates with a graph facade. They cover all requested sizes, a deliberately
non-writing eager empty kernel, a graph-replay-only non-writing kernel, and a
fallback masquerading as native. Both corruption controls fail exact-zero gates.
The facade records in-graph output poisoning and extension calls. CPU fixtures
use three replays for bounded host cost; the real probe defaults to 2000. These
tests are host scaffolding evidence, not a substitute for CUDA graph tests.

Logs:

- `muse_moe_w4a16/profiles/fork_moe_empty_source_cpu.2026-11-01.log`
- `muse_moe_w4a16/profiles/fork_moe_empty_torch_cpu.2026-11-01.log`

## Next H200 conformance and stopping criteria

`tools/dl_fork_moe_empty_probe.py::empty_conformance(torch,state,selector,
native_factory,fallback_factory)` is CHECK ONLY. Invoke after full preparation
using the candidate's `_moe`, `_fork_native_moe`, and a private prepared control.
Root may supply official Marlin directly instead of the incumbent fallback.
The probe fails closed unless the selected M32/33/128/256 modules and M1 helper
are the actual native modules. It exercises B4FIN and main, plus helper/native
interleaving; upstream global-ID-to-local transformation is outside this direct
local-ABI test.

For each size: first prefill output with 17 and call all IDs=-1, requiring finite
exact BF16 zero; create four fresh input banks; compare an occupied native call
against the control; capture occupied -> empty -> M1-empty using the same native
scratch; re-poison both empty outputs INSIDE every replay immediately before
their launches; queue 2000 replays while rotating fresh input bank copies. Three
launches per graph exercise both launch parities. Only the first eager empty
result and final replay outputs are explicitly observed; no claim is made that
all 2000 intermediate outputs were individually checked. Final occupied result
must remain finite with global relative L2 <=.008 and max row relative L2 <=.075;
both final empty outputs must be finite exact zeros. These are control comparisons,
not an independent FP32 mathematical oracle. Official layer correctness remains
mandatory for numerical qualification. No threshold relaxation is authorized.

CUDA launches, captures, transfers, synchronization failures and budget violations
are fatal. The 280-second probe guard checks between cases, not an interruptible
hard deadline or a bound on total prepare time. Keep graph/probe code off pure
files. Any fallback selection, nonzero empty result, nonfinite output, correctness
failure or graph issue blocks native release; revert to the independently
qualified fallback rather than infer safety from a normal routing pass.

## Separate unresolved per-token finalization design

This empty repair does not enable MFIN. Existing bounded readiness polling can
fall through without observing completion; enabling its flag alone is forbidden.
Old owned mode79 (`2a4fb39964aa448c`, recorded in runs/ruled-out) lost 8–15us
against its phase-B claimed control. That experiment changed stream-K phase-B
fixup ownership/walk, not this native finalizer's global rendezvous. It neither
proves nor closes a distributed native finalizer.

A possible safe design retains fixed token/segment output owners and adds
per-expert completion publication after every phase-B writer (including split
finishers). Each owner acquires all local experts' exact completion targets before
reading their y/SS outputs. Every contributing WG must bridge its own stores
through the existing WG synchronization/proxy/release protocol; a release by an
unrelated lane alone is insufficient. Replace bounded poll fall-through with a
readiness loop whose only exceptional exit traps/rejects. All-resident cooperative
launch is necessary for progress and must remain independently checked.

The old start-ticket epoch depends on fixed grid size and shares generation state
with skinny helpers. A more robust separate native epoch would toggle only after
the last CTA retires its final reads, use a generation-safe last-exit counter,
and reset the next parity before reuse. It also needs separate grid-barrier
counters in diagnostic control arms: native epoch parity cannot silently select
the helper's unrelated `ctr[3]` parity. Reserved MFIN storage has CE 0..63,
exit at64, XR128+, DA384+, but adopting new epoch/barrier offsets requires a full
layout/reader audit and CPU interleaving models, not simply assuming gaps are free.

No finalizer patch is provided: publication coverage, graph interleaving and
retirement proof remain substantial dependencies, and no measured exposed
finalization opportunity exists on the repaired native path. The smallest useful
next experiment is an attribution-only twin recording each expert's last phase-B
publication, each fixed owner's phase-B finish, and final read/retirement on held-out
routed seeds at128/256. For token t, earliest unchanged-owner start is bounded by
`max(owner_phaseB_finish, max(publication[e] for local routes of t))`. Compare
critical completion to the actual global-rendezvous path including unchanged
normalization time. If most tokens depend on the globally last expert, eliminating
the grid barrier may expose little gain. No projected speedup is justified yet.

Only after that ceiling and a complete publication/epoch model should a future
2x2 separate epoch policy from readiness policy, with both control combinations
proved valid, independent duplicate scratch, unchanged BF16/SC6/ABI, resource
equivalence, fresh replay and empty-routing coverage. No checks or G1b changes
are authorized by this design note.
