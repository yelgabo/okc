# Independent G stage-probe review — device checkpoint

Evidence base: reviewer commit `4367810`; inspected live MoE worktree on
2026-11-01. No H200 job, production change or dispatch clearance in this checkpoint.
This is an instrumentation review, not a release or performance assessment.

## Exact source and compiler evidence

`tools/dl_g_stage_probe.py` raw SHA256
`9f85088a154bf6c55b55ec90189a8336571c1725447d9f07bc26c59f27eab014`
pins the original device body to
`c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`.
Inspected generated `g-stage-probe-v4.cu` raw SHA256
`311f874cc70cd69c3028556f42f2c5f2098296af57bb2f8711cc60b8b4b85402`.
Both are in `../okc-today-moe` relative to this checkout.

Read the actual local CUDA12.8/sm90a manifests, PTXAS logs and SASS under
`/root/.cache/okc-g-stage-{ready,retire,endpoint}-v4/`. The relevant options
include `OKC_NW=16`, `OKC_FAST1=0`, `OKC_WGS=1`, `OKC_NS=3`, `OKC_NSF=3`,
`OKC_PRMTSC=1`, `OKC_SC6W=1`; the source disables MFIN in this family.
NVRTC compilation excludes the Torch host suffix and uses disclosed virtual
standard headers plus a forward-declaration adapter. It does not certify the
real Torch host extension or GPU execution.

| Build | Registers | Barriers | Stack | Spill stores / loads | HGMMA / wait1 / wait0, both entries |
|---|---:|---:|---:|---:|---:|
| READY | 96 | 16 | 8 B | 44 / 68 B | 1024 / 512 / 96 |
| RETIRE | 96 | 16 | 8 B | 4 / 32 B | 1024 / 512 / 96 |
| ENDPOINT | 96 | 16 | 8 B | 4 / 32 B | 1024 / 512 / 96 |

Counts were independently recomputed from actual cached SASS. READY has real
additional spilling; equal register counts alone do not establish low distortion.
Endpoint's manifest pins source `0531e486...`, unlike READY/RETIRE's
`311f874c...`. The owner has been asked to reconcile that source-version
difference before presenting these as one frozen family.

## Device safety findings

No device blocker found in the inspected body:

- The original `Smem::fcnt` consumers/writers are in the FAST1 branch, excluded
  by actual `FAST1=0` and non-tiny main flags. The observer reuses only its first
  three words; neither shared layout nor allocation size changes.
- Phase setup writes a split 64-bit trace pointer and sequence origin. Only
  thread0 later loads them: PTX branches other threads out before shared loads.
  Scoped temporary registers avoid cross-asm naming collisions. The asm has a
  memory clobber, and no cross-thread publication is needed for this private
  state. Source lifetime reaches the next phase setup without a native fcnt use.
- Original waits, WGMMA drains, proxy fences and counted arrivals remain.
  READY adds nonblocking acquire tests; it does not replace the original waits.
  Readiness addresses match actual `BW_F=0`, `BX_F=6` and three-slot rings:
  `(seq%3)*8`, then `+48` bytes, with parity `(seq/3)&1`.
- Timestamp rows occupy words0–167; phase headers168–203; consumer finish204–206;
  job count207; at most88 job triples208–471; CTA entry472–474. The declared
  475-word buffer covers all accesses. Selected CTAs are0, grid/2 and grid-1;
  supported geometry must keep these distinct. Consumer finish is inside the
  existing `tid==0` block after CTA synchronization.
- Job metadata is read after routing's existing synchronization. The CPU
  reconstruction matches native fat-first/expert-order chunks64 and skinny
  jobs, and phase-A trailing partial ordering. Sequence numbering continues
  across phases, while selected ordinals restart relative to each phase.

## Interpretation and remaining gates

These are sparse leader observations under the original weight-first wait
order. X wait after weight excludes any overlapping X readiness delay already
hidden by weight waiting. Snapshot tests themselves precede mark0's clock.
RETIRE brackets the full WGMMA drain and subsequent release path, including
observer overhead; neither interval is a hardware bandwidth/copy-duration
measurement or an additive whole-layer saving. Phase end includes pending
flush; consumer finish precedes the grid barrier/final normalization.

Before source/API clearance, review the final generated host artifact and require:

1. A private five-element scratch ABI for instrumented arms, with the trace
   contiguous CUDA int64[475] on the input device; no accidental four-element
   native call or parameter-layout mismatch.
2. Trace zeroing on every replay. READY writes only the low32 bits of its
   readiness word; untouched high32 bits and unused rows must remain zero.
3. Six arms with independent scratch and matched inputs: two uninstrumented,
   two endpoint, READY and RETIRE. Actual compiled geometry/admission and
   resources must be queried, not inferred from local device compilation.
4. Correct natural routed outputs/fresh replay state at MAIN128/256, metadata
   reconstructed from the exact replay IDs, complete required rows and a
   fail-closed transport/deadline guard.
5. The agreed <=1.5% endpoint/null and <=3% detailed whole/phase distortion
   gates. Failure closes causal use of these stamps; it does not establish a
   kernel bottleneck or justify modifying production.

Next action is final host-artifact review, then root alone decides whether to
dispatch the bounded diagnostic. No new mechanism or quota request is proposed.

## Subsequent host-draft review

Inspected `g-stage-check.py` 391252 B, SHA256
`2703fe825a3416e069698ff13b78afec3a3c7f7a88d0505367f6b64004d649a7`
and `moe_make_g_stage_probe.py` / `moe_g_stage_body.py`. The scratch construction
is correctly three original buffers plus an unused placeholder and trace. Mutable
scratch/output/trace allocations are checked for storage aliasing. Instrumented
traces are cleared before timed and decoded replays; decoding follows explicit
replay and synchronization. Natural routing, fresh values, skew, all-remote and
restored inputs are exercised at both128/256 with three seeds.

The following were sent immediately to the owner and coordinator, so this draft
is not yet cleared:

- `_gsp_errors` tests output finiteness only. A nonfinite reference can produce
  NaN error metrics, making both `>` rejection comparisons false. Require
  finite reference and finite metrics, with a CPU negative test.
- Budget checks occur before cases, but not inside the150 timed laps/case,
  snapshot loops or after the last case. Require fail-closed inner/final guards.
- The transport assertion allows14000 characters. Require an executed realistic
  six-case report plus four475-word traces fixture against the confirmed retained
  log limit; the earlier diagnostic practical budget is11KB.
- The `.015` global parity tolerance follows the old SC6 harness but must be
  described as layer-style parity, not the literal standalone `.008` test gate.
  The normal official control certification remains separate.

The owner explained the endpoint source-version difference as an uninstantiated
helper's PTX predicate repair, and is rerunning endpoint/control with the exact
current source rather than claiming those raw hashes are identical. Final CPU
host facade/negative tests and frozen artifact identity remain outstanding.
