# A smaller BF16-matched selected-cell comparison

November 1, evidence starts at 5f5098b, with the hourly direction in a63588b.
The prior turn made progress through two official H200 measurements and their
closures. This continuation implements a new comparison package and sends an
independent attention wait experiment to H200. The public incumbent remains
607eb52bdaf24160 at 2.126083×, rank 305 at the start of this work.

## What now fits

The fixed verify-32 sliding comparison packages at **850756 bytes**, SHA256
6cf6a3a701be05e7ac8e8b619b4e5b7f6a2c68131aa1f15ce2190a7ee0113492.
Windows and WSL generate identical bytes. It contains two explicit private
implementations, the independently restated FP32 judge reference, and SGLang's
normal layer as the official judged entry. The module sources are readable
multiline Python strings with an in-memory source cache for Triton. There is no
encoding, compression or external source access in the payload. This is a
private diagnostic. [Generator](../../tools/dl_selected_pair.py).
[Manifest](../../muse_decode_layer/profiles/main-slice.2026-11-01/pair-manifest.json).

The owned arm retains the pinned BF16 attention-output repair. It now omits four
complete skinny MoE namespaces that its fixed launch_main<51> dispatch cannot
reach. Removing them saves 108254 bytes. The selected SM90a kernel has all
**17528 printed SASS instructions identical** in the original and reduced builds,
with 96 registers, 16 barriers and zero stack or spills under CUDA 12.8. Only host
launch wrappers are excluded by the local compiler adapter. Real extension host
linking and H200 execution remain untested.
[Owned specialization](../../tools/dl_owned_main_slice.py).
[Compiler equivalence](../../muse_decode_layer/profiles/main-slice.2026-11-01/equivalence.json).

The G arm is 406300 bytes and keeps the incumbent's entire ATT7 and native MoE
device literals byte-for-byte. Its explicit selected-load preparation creates
the four pinned Lt plans, original gamma and residual coefficients, native SC6
weights, guarded scratch and the normal attention workspace owner. Its pipeline
matches the selected source order, including both rounded BF16 boundaries and
the FP32 pooler/router concatenation. It retains the native cooperative launch
checks. Unrelated small-token families and preparation are omitted.
[G slice](../../tools/dl_g_verify_slice.py).
[Independent semantic review](2026-11-01-g-verify-slice-independent-review.md).

## Corrected host guard and ownership

The earlier owned slice incorrectly required an 80000-column page table. The
actual agentic 80000-context input reserves the longest supported request plus
eight draft positions, giving **86008 columns**. The corrected test executes the
real input generator's metadata construction, replacing only cache payload
allocation and the optional FA3 scheduler. The slice has never been dispatched
on H200, so this error affected the offline prototype. Both slices now reject
the unsupported capacity. [Metadata regression test](../../tools/test_dl_owned_slice.py).

The owned Lt functions capture their workspaces in closures. A Python deep copy
would share them. The harness instead prepares a second owned state and rejects
workspace or scratch overlap. G uses the existing conservative owner clone,
which preserves packed read-only weights and copies Lt workspace and scratch
backing storage. A test demonstrates that the unsafe closure deep copy is
rejected. All four input/cache banks are separate. Each selected attention path
must run eagerly twice before capture so that its actual workspace exists.
[Runtime](../../tools/dl_selected_pair_body.py).
[Packaging and ownership tests](../../tools/test_dl_selected_pair.py).

Three G slice tests, four owned slice tests and three package tests pass.
The literal module loader also executed a small Triton kernel on the local RTX
5070. That validates the in-memory source mechanism. It supplies no H200 latency
or native layer correctness result.
[Local loader fixture](../../muse_decode_layer/profiles/main-slice.2026-11-01/loader-cuda-probe.py).

## Required H200 decision

This packaging result supersedes the whole-literal size obstruction for this
single cell. It does not establish the six-load cross-family comparison.
The 300-second combined setup budget, actual source linking, selected plans,
native resources, guarded captures, math and naturally identical expert IDs
remain required H200 gates. Routing differences abort the private comparison.
Two original and two fresh held-out banks must complete all 24 oracle checks.
The official baseline must then pass all 13 cases.

The output reports only verify-32 sliding time ratios. It cannot claim a complete
score gain. Promotion still requires a compatible, measured twelve-cell result
under the 46/16 workload weights. Keep the complete legacy-pair closure intact.
The registered reset work remains the incumbent stage comparison and short-path
trace. This package can be considered after the remaining host and ownership
review. No layer check or public attempt was consumed by the packaging work.
[Legacy scope](2026-11-01-layer-legacy-pair-closure.md).

## Attention job running

Private job 3f44f891d0ef47db measures sparse producer and consumer waits in the
owned full-context attention at batches 16 and 32 for decode and verify. The
361621-byte source has SHA256
a882bd2ad7d9ebf36a5d4f739e6c0b60930a6e733e24fed06712090bc751d9c0.
Two independent detail builds and two unstamped controls share math and input
values, with private caches. Exact register, local memory, shared memory,
thread and residency parity is required. Six CPU tests pass, and the
independent review cleared the source with explicit clock interpretation limits.
Only rows inside the registered perturbation gate can support a bottleneck
decision. [Experiment](2026-11-01-attention-full-wait-trace.md).
[Independent review](2026-11-01-attention-wait-independent-review.md).
[Dispatch](../../muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.dispatch.json).

The MoE owner is preparing a separate phase A/B experiment using the actual
incumbent stage backend. It separates observed weight/X readiness, stage cadence
and retirement, and uses endpoint controls to bound instrumentation effects.
MAIN at 128/256 is the initial scope. B4FIN at 32 remains unmeasured.
