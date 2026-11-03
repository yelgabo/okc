# Internal MoE profile validation

November 2, 2026 Pacific. User requests internal MoE timings with an unchanged control and measured instrumentation distortion. Checked repository base `742c0a77` and the MoE worker's uncommitted observer snapshot. This is a local methodology audit, not a completed GPU profile.

The current Mac and its Linux ARM64 Docker runtime have no NVIDIA driver or GPU execution. Local runs can validate source transformations, compile device code and exercise synthetic host logic. They cannot determine H200 phase costs. The existing MoE worker owns the target diagnostic. The user explicitly authorized messaging that chat; the profiling request and the failure finding below were delivered there. This chat starts no duplicate job and changes no worker files, kernels or reminder.

## Selected diagnostic and control

The worker's generated payload was `b436704cb2469235b3c94ca304f58d57e416d704282430c24f7a6d1d3223202c`, 351236 bytes. It retains native control `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`. Its official arm returns uninstrumented control0. Diagnostic arms add sparse boundary timestamps through the existing debug pointer. This source identity describes the reviewed snapshot, not any later accepted upload.

The design uses two original controls and two observed copies, independent mutable storage, fresh routing, normal and skewed distributions, three input banks, graph replay and shuffled forward/reverse timing order. It plans 32 repetitions per arm for each timed case. Internal trace coverage is T32, T128 and T256. Tiny and MID internal attribution remains outside this observer.

The snapshot rejects complete-call displacement or duplicate disagreement above `max(0.5 us, 1% of control)`. This is a coarse perturbation screen. It does not prove small intervals are accurate or that instrumentation has preserved the internal schedule. Retain variability and repeated trace agreement, and report the uncertainty alongside any accepted interval.

| Signal | Supported interpretation if target gates pass |
|---|---|
| Entry through routing/gather | Startup span including synchronization |
| Per-warpgroup A completion and B completion | Work and waiting within each observed group |
| First B readiness check | One sampled dependency wait per CTA, not all readiness cost |
| CTA completion spread | Measured tail spread, not automatically removable time |
| Last compute through final exit | Join and finalization exposure |

Weight delivery, register decoding and WGMMA overlap in this implementation. These boundary stamps do not isolate each one's active duration or produce additive percentages. Phase B can begin before every CTA finishes phase A. Keep that overlap visible in the diagram overlay.

## Local findings

Three source tests and two synthetic host tests passed independently. They cover selected source/flags, reversible instrumentation edits, malformed trace rejection, independent host preparation, replay counts, official control selection and a guard failure inside preparation. Their timestamps are synthetic. They establish no GPU correctness, overhead or timing result.

The worker's saved local B4/MAIN compile logs report 96 registers and 16 barriers for both control and observer entries. Existing B4 helper spill records also match. These are compiler observations, not target resource or launch-admission checks. No new compiler run was needed to repeat those saved observations.

One additional synthetic failure check reproduces lost evidence. After successful cleanup, the inherited budget check raises outside the partial-report handler. No `MOE_PHASE_PARTIAL` record survives, despite completed observations. The worker received the exact failing path. Reset errors and final report/budget failures also need bounded partial reporting with the original failure preserved. The [reproducer and output](evidence/2026-11-02-internal-moe-profile-review/check_post_cleanup.py) are CPU-only.

The CUDA reference checker reports a changed `docs/ruled-out.md` pin. Its affected notes need reconciliation by their owner. This audit does not refresh hashes or use the stale reference as performance evidence.

## Target completion and stop

The MoE worker should finish its existing preregistration and failure handling, recheck jobs/quota and run one bounded private observer check. Keep exact source/build/packing/dispatch identities and the unchanged official control. Validate correctness, replay, ownership, target resources and trace ordering before interpretation. Preserve partial evidence on every failure or budget exit.

Publish the accepted job, raw receipt, control timings, observer timings, duplicate noise, uncertainty and permitted phase spans. If distortion or stability fails, reject attribution and stop that observer. A passing complete-call screen permits only claims consistent with the trace's resolution and sensitivity checks. A split of pure memory, decode and tensor-core time would need further evidence on an accessible H200 profiling host.

No target job is claimed by this report. The actual internal profile remains pending with its existing owner. The [local evidence summary](evidence/2026-11-02-internal-moe-profile-review/summary.json), [source manifest](evidence/2026-11-02-internal-moe-profile-review/paired.manifest.json) and [file/resource hashes](evidence/2026-11-02-internal-moe-profile-review/local-review.json) bind this audit. The older intrusive observer remains invalid as recorded in the [phase brief](2026-11-02-cuda-b4-phase-attribution-task.md).
