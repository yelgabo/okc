# Projection async-mailbox scope: no new port justified

2026-11-01. Evidence base `ca3ec15`; C2 job `9eba96894ee1402e`, raw
source SHA256 `5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4`.
Old owned QKV push job `4edaa20c98904404`, source breadcrumb `aded9cf`.
No kernel changes, GPU dispatch, submission or projection quota request.

**The proposed async-mailbox transport is already present where inter-CTA
reduction is active.** The short SK/RD kernels selected by C2 reduce inside
a CTA, so describing them as global reducers incorrectly enlarges the scope.
No newly identified exposed reducer budget supports a next-reset check.
This closes the proposed direct port, not all future epilogue scheduling.

## Actual selected paths and ordering

Exact source is preserved in the prior compiler audit's
`evidence/2026-11-01-projection-waits/device.cu`; exporter configuration and
all22 actual lowered entry/resource reports remain beside it. Those are
local CUDA12.8 compile results, not H200 measurements. Root separately
owns the CUDA13 compiler comparison; no new compilation was needed here.

| Active family | Selected C2 cells | Actual completion/reduction |
|---|---|---|
| SK | qkv8/16/32, O16/32, pool16/32, unpool8/16/32 | All selected configurations COMP=true, CL=1, FUSE=false. CTA-local warp partials overwrite the dead ring, named barrier before and after stores, local FP32 sum and one BF16 output rounding. |
| RD | pool1/pool8/O8 | CTA-local `red[warp][nt][lane]`, CTA barrier, local FP32 sum and BF16 output. No cluster/remote partial transport. |
| RT | router8/16/32 | Already dedicated receive buffer and remote `st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32`; receiver `mbar_wait_cluster` is acquire.cluster. Own-rank share is a normal local store followed by CTA publication. Selected TAIL3 splits the exit arrival/wait around owner reduction. |
| WD QKV128 | one cell | Shipped control aliases dead ring for FP32 partials, cluster publication/join, generic DSM reads, BF16 output, then lifetime exit join. The losing owned QKV push experiment already tested a dedicated async mailbox alternative. |
| WD router128/256 | two cells | Already async-push paths (VAR6190), not a fresh DSM-to-mailbox target. |

Source anchors in the persisted device CU: SK flags/configuration248-310,
local partial/reducer794-858; RD2281-2311; RT2032-2095; WD helpers1337-1340
and reducer1721-1815. The compiled SK template CL argument is1 in every
selected entry, so its dormant remote and fused/global branches do not
describe the incumbent. RD also runs pool1, not merely its named M8 route.
M1's four other CUDA-core GEMV families and seven large-M Lt cells are
outside this mailbox hypothesis.

The old `tools/proj_make_bench_qkv_push.py` is decisive: its EPILOGUE calls
`wd_st_async(...,wd_mapa(rbar,dst))`, waits on incoming transaction bytes,
then reduces local received FP32 partials in original rank order. Own
partials remain in registers. Its separate receive region grows shared
memory197760 to222352 bytes; the padded BF16 transpose aliases the dead
ring. It is therefore **already an async complete_tx mailbox**, not a
generic DSM-read implementation. The receiving wait has cluster acquire,
not merely a CTA-only acquire. One may not substitute a weaker acquire
scope while asserting the same remote-publication proof.

The initial cluster join still orders ring/remote-arrival lifetimes; the
terminal join prevents exit with peer transactions in flight. Transaction
completion on an output mailbox does not establish every earlier TMA or
remote empty-barrier completion. The repository's 2026-10-30 conforming-path
decision (`docs/runs.md`, section1117) explicitly retains exit lifetime
joins and avoids relying on st.async targeting its issuing CTA. No fence
or join removal follows from the VP2 analogy or successful math tests.

## Actual receipts and score sensitivity

Verified old push receipt owns board `muse_bf16_proj`, job4edaa20c98904404,
passed=true. Run-log paired observations: push versus independent rebuilt
pull **+0.352 to+0.480us**, versus production **+0.272 to+0.416us** on two
heldouts/two builds. All FP64, one-hot, nearby-M and three fresh-graph gates
passed;90 registers/zero local/one CTA/39 active clusters were unchanged.
This is a performance loss, not a compile/correctness closure. Existing
`docs/ruled-out.md` row298 already closes that exact schedule, retaining
distinct scheduling/real-component evidence as reopening triggers.

`tools/proj_mailbox_scope.py` reproduces full30-cell accounting from the
same C2 receipt audit, retaining every cell's official weight62. Six load
sums are35.182617/37.668069/46.914076/51.720988/67.095785/82.416369us;
the original full score reconstructs1.5278064829557674. The independent
output is `evidence/2026-11-01-projection-mailbox/scope.json`, including
receipt hashes, all30 rows, and explicit null reducer measurements.

| Uniform hypothetical whole-operator cut | Cells | 0.5us each factor | 1us each factor |
|---|---:|---:|---:|
| SK |10|1.018487|1.038068|
| RT |3|1.005648|1.011394|
| RD |3|1.006894|1.013996|
| SK+RT+RD |16|1.031708|1.066380|
| all WD |3|1.003523|1.007103|

These are score sensitivities, not measured reducer costs, predictions or
attainable bounds. Each profiled operator is one kernel; receipts provide
no reducer-only timing. The 1us-per-SK scenario would have material score
leverage, but there is no evidence that one exposed microsecond exists in
each local epilogue, or that a remote mailbox can remove it. The WD-only
1us scenario is much smaller. Do not infer a top50/top10 gain from these
counterfactuals or combine them with unrelated wins.

## Decision and reopening criterion

No new kernel or check is justified by renaming the existing transport.
Preserve C2. Stop this async-port proposal without a tile/stage/hint sweep.
Reopen only with a distinct legal local reduction/ownership schedule and
actual exposed component evidence (or an instruction/lifetime proof of a
redundant critical serialization), plus a full30-cell score budget. A
dedicated local partial buffer could alter the first ring-reuse barrier,
but costs extra shared memory and has no demonstrated time/admission
advantage; it is not an implemented or check-ready proposal here. Any
future removal of ring/exit joins requires an explicit memory/lifetime
proof independent of test results.
