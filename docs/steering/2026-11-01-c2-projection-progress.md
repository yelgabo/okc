# Retained C2 projection: scoped progress eligibility audit

2026-11-01. Reviewer base `59a3901`; no kernel changes, GPU execution, official
calls, quota request or submission. This follows the owned MoE progress defect;
it does not transfer that defect to kernels with different synchronization scope.

**No active SK/RD/WD whole-grid readiness join requiring whole-grid residency
was found.** The retained C2 recommendation is unchanged by this bounded audit.
This is a source-level progress determination, not a new numerical, memory-model,
vendor-library or performance certification.

## Exact source and owning run

Restore with `tools/proj_sk_xor_control.py::restore(root)`: tracked source blob
`3044ee7:muse_bf16_proj/submission.py`, authorized repeat comment and CRLF.
Raw source: 216576 bytes, SHA256
`5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4`.
Normalized LF SHA256:
`7f4fd58cdd940f36c8d4d355f3de85687bc7f6ab04981842ce936db76d78adc2`.
The line anchors below refer to that restored file, not current tracked C3.

Owning receipt `muse_bf16_proj/profiles/scored.9eba96894ee1402e.json`:
job `9eba96894ee1402e`, board `muse_bf16_proj`, PASS9, all30 benchmark cells,
score **1.5278064829557674**, 95% CI
[1.504889385711431, 1.5507235802001038]. Correctness pass is recorded evidence;
it is not the scheduling proof. The already committed full30 receipt/profile
mapping is `evidence/2026-11-01-projection-waits/audit.json`.

## Selected paths and progress scope

| Path | Actual selected cells | Wait/reduction and launch |
|---|---|---|
| SK10 | qkv8/16/32, O16/32, pool16/32, unpool8/16/32 | COMP=true, CL=1, FUSE=false. Producer/full/empty barriers, escape readers and WSK warp reduction are inside the same CTA. Named barriers protect local ring reuse and local partial publication. `launch_sk`1153-1221 takes ordinary CTA launch for CL1; optional PDL does not supply or require a grid rendezvous. |
| RD3 | pool1/8, O8 | `rd_kernel`4042-4158 uses shared `red[NWARP][MT][32]` and CTA barriers. Each CTA owns different output rows; WK is a warp partition within the CTA. `launch_rd`4160-4173 uses ordinary CTA launch. Escape-list iteration reads immutable records, not another CTA's readiness flag. |
| WD QKV128 | one cell | Actual template `wd<1,128,3,1,false,8>`, ABL0/POLtrue/VAR0. Three-CTA cluster. Shared ring publication, DSM partial reads and exit lifetime synchronization are cluster-local (3110-3199). |
| WD router128/256 | two cells | `wd<1,64,8,1,true,8>` and `wd<1,128,8,1,true,6>`, ABL0/POLfalse/VAR6190. Eight-CTA clusters; async partial pushes target only ranks in the same cluster. Receive acquire and split exit lifetime join are 2951-3104. |

VAR6190 is PACK(2)+PROMO64(4)+PUSH(8)+XA(1024)+TAILSPLIT(2048)+
TAILRLX(4096). It does **not** set GRED(65536), NOEMPTY(512), NOPROD(1),
or the diagnostic SPIN(16). Confusing this selected path with GRED would give
the wrong progress proof.

`launch_wd`3220-3273 uses the same `C::LCL` as the native rank/cluster logic,
sets `cudaLaunchAttributeClusterDimension`, and checks `cudaLaunchKernelEx`.
For these selected templates, LCL=CLK*CLR is3 or8; GRED is false. Dynamic-shared
attribute setup is checked. There is no `cudaLaunchCooperativeKernel` or explicit
all-grid occupancy admission here, and neither is needed for the cluster-local
joins. An invalid cluster launch must fail the checked CUDA launch rather than
execute as independent ordinary blocks. This audit does not assert a measured
active-cluster count or absence of resource pressure.

NVIDIA specifies that blocks within a thread-block cluster are co-scheduled on
one GPC; different clusters need not all reside simultaneously. That is the
required scheduling guarantee for the selected WD waits. See the primary
[CUDA programming guide, Thread Block Clusters](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#thread-block-clusters).

The remaining custom inter-CTA short path is RT router8/16/32, already audited
in `2026-11-01-projection-mailbox-scope.md`: rank-local DSM async mailboxes and
cluster acquire/lifetime joins. `launch_rt`3543-3575 likewise declares its S-CTA
cluster and checks the launch. The four GEMV short cells have no readiness
join; the seven Lt cells introduce no owned software grid join. Vendor Lt
implementation internals were not independently proved by this source audit.

## Dormant branches and limits

SK FUSE has a global counter/reducer branch (1088-1091), but all selected SK10
use FUSE=false. WD GRED (3130-3170) is also unselected. Both are written as
last-arriver reductions rather than a resident CTA spinning for future CTAs:
earlier CTAs can return and release resources. This distinction rules out the
specific whole-grid residency argument; it is not an independent endorsement
of dormant branches' arithmetic, publication or reentrancy contracts.

No fences, named barriers, cluster joins or reader counts were removed. Existing
WD push and pull lifetime joins remain necessary for their own memory-ordering
and shared-storage lifetime proofs. Watchdog success and historical test passes
were not used to infer safe progress.

## Next gate and stopping criterion

Close this eligibility question without a new check or production patch.
Retain exact C2 as the measured projection candidate over the two losing sparse
representations. Reopen only for a concrete selected source change introducing
cross-CTA readiness outside the declared cluster, an actual flag/launch mismatch,
or a source-derived progress counterexample. Such a change requires a fresh
ownership/lifetime and scheduling proof before official validation. This audit
does not predict a rank, a speedup or a top-10 cutoff.
