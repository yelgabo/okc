# G0→607e native minimal-delta ladder

Source/CPU audit only, November1. Reconciled root `0f51f77` and subsequent human
priority `ccb5e19`; own preceding handoff `be42839`. Campaign remains paused:
no GPU, official calls, queue changes or production edits. README is left to
root by explicit instruction. The machine-readable ledger is
`evidence/2026-11-01-g-native-minimal-ladder/inventory.json`.

**New result:** applying the existing pinned `dl_fork_moe.patch_device` to the
exact G0 helper0 reconstructs607e helper0 **byte for byte**, SHA
`b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716`.
For main, the same repair plus existing `dl_g1_stage_retire.patch_device` and
the80-byte `wgs_stage_at(nj==0)` guard reconstruct607e exactly. There is no
unclassified native-literal remainder in this comparison. This is a source
identity result, not a causal latency attribution.

## Pins, methodology and actual dispatch

G0 is upstream `8e30e1be994db4310794b8d04e90d18d895deb84`, layer LF
`82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`,
owning check `3a5f879df0d44825`. 607e layer LF
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`
belongs to public `607eb52bdaf24160`. Author/team board scores are not substituted.
The ledger pins the complete repair/stage/empty tools and each original/final
literal, with source-assignment lines, every ordered tool-line edit, native
line hunks and before/after hashes. Apache-2.0 upstream attribution remains in
the owning source; no kernel is newly imported or dispatched here.

`tools/g_native_patch_ladder.py` instruments **the existing** transformer's
string assignments:21main and12helper checkpoints,12group classes. It asserts
the uninterrupted hash chains terminate at the exact incumbent literals.
Partial checkpoints are not conforming or runnable ablation candidates; many
are intentionally inconsistent until their coupled group is complete. It also
executes only the pinned pure-string `_df_patch` definition to pin original
MID's actual derived source, not the submission or any CUDA code.

Two CPU tests pass: exact full chains/no remainder and actual configuration/
gamma-consumer scope. Reproduce with `OKC_REPO_ROOT=ROOT python -m unittest
discover -s tools -p test_g_native_patch_ladder.py -v`. Generate the JSON using
`python tools/g_native_patch_ladder.py ROOT OUTPUT`. No new compilation was
needed to establish this source identity.

|tokens|selected native source/flags|G0→607e scope|
|---|---|---|
|1|helper0 Tiny, WS1, WSBL2, WSBLK2, FINPF1|short WS numerical/publication/launch changes; no WGS MFIN/YBF/SC6|
|8|helper0 Tiny2, WSBLK1, CPF0, FINPF1|same repair families, distinct retained shape flags|
|16|MID SPLIT1, helper0df name|G0's separate DF sentinel transform replaced by alias to repaired helper0|
|32|main WGS+B4FIN1|main-specific fixes plus stage retirement and empty guard|
|128/256|main WGS|same main fixes, ordinary finisher, eligible SC6 retained|

The **final actual tuples**, including helper-source suffixes and MID alias,
are identical between sources in the ledger. PDL/MOE_HAS_PDL/MFAB_LAYER differ;
SC6_WGS remains true. Main WGSFCI4 remains the active main value. Helper's
compiled-out WGS block/old CI3 comment does not change that selection.

## Ordered ladder: requirement versus representation

|order/group|required property versus chosen implementation|safe minimal isolation|
|---|---|---|
|1 scope guard/MFIN0|Exclude unreviewed fallback; unsafe MFIN bounded-poll fallthrough excluded. MFIN affects B4/main, not short WS.|No flag rollback. A new finalizer needs its own publication/epoch/progress proof.|
|2 BF16 primitive|Adds the conversion helper definition; definition alone has no established runtime cost.|Not an independent performance hypothesis.|
|3 effective gamma|Baseline materializes BF16(1+rawBF16). Current per-use `fork_bf16(1+raw)` is a representation of that required value.|Preparation-time immutable effective coefficient is mathematically possible; host and **all** native consumers must change together.|
|4 gate/up boundary|Scaled GEMM1 gate/up must round before nonlinear SiLU.|Keep boundary. An unrounded negative/control arm would not be conforming.|
|5–6 down boundary/scale separation|Scaled GEMM2 rounds before gamma; WGS global scale must be outside gamma at this edge.|Treat changes as a coupled numerical group. Moving multiplication through BF16 is not equivalent.|
|7 YBF0|Extra postgamma quantization removed conservatively, distinct from required pregamma round.|No proved rollback or isolated price. Main/B4 only.|
|8 skinny reader bridge|Generic reads finish before async ring refill; existing per-reader proxy fence precedes compute join.|No fence removal on timing tests. Active short WS path, not merely unused code.|
|9 WGS release counts/fences|Every reader's release and matching NT expected count form one protocol.|Counts and reader participation cannot be ablated separately. Main/B4 only.|
|10 WGS writer bridges|Generic xA/xB stores, including newly produced split-finisher stores, publish to TMA.|Entry fence alone does not cover later stores. Main/B4 only.|
|11 final-grid RMW strengthening|Writer/CTA relay, GPU fence, strong counter RMW, acquire and final join are required. Added acq_rel **on this RMW** is one conservative representation.|Only short Tiny/Tiny2 have a scoped alternative proof and selected compile so far; retain every explicit fence/round/join.|
|12 cooperative/PDL policy|Software-grid progress requires admission. Checked cooperative launch supplies it; PDLoff same-stream completion avoids unfinished-primary residency dependence.|A different admission/dependency policy needs its own complete proof, not a PDL boolean toggle.|
|13 stage retirement|Strengthened per-current-X retirement recovers compiler wait1 overlap while preserving reader lifetime.|Already in607e; do not count it as future recovery.|
|14 empty guard|No job metadata access at zero local routes.|Required bounds repair; no old uninitialized empty arm.|

Host-only state-local scale/packing/cache ownership, MFAB diagnostic removal,
explicit Marlin fallback and eligibility checks are separate from the literal
ladder. They prevent format confusion or invalid path selection; no isolated
timed cost has been measured. SC6 is retained where losslessly admitted, not
a removed optimization.

Required BF16 operator boundaries are justified by the pinned official baseline
and SGLang contract, not numerical tolerance passes. Publication and progress
obligations follow the scoped primary references in the existing
[precision contract](2026-11-01-g-fork-precision-contract.md) and
[ordering restoration proof](2026-11-01-g-ordering-restoration-proof.md).
This does not convert a conservative internal-precision choice into a universal
challenge rule or authorize dropping a proxy fence.

## What the short counter check actually isolates

The f042 harness is a **conforming alternative to one added implementation
choice**, not an ablation of required ordering, and not exact G0. It replaces
the strengthened final-counter acq_rel RMW with an explicit returning relaxed
strong GPU RMW. All preceding607 numerical and protocol repairs remain.
Thus it is related directly to the repair ladder; it is not an unrelated kernel
optimization. A future win would measure that choice *conditional on current
607e*, including compiler interactions, not the cost of every safety repair.

The scoped proof covers compute-tail stores relayed through full-NTL WS_JOIN
to leader GPU fence, the same-location RMW observation chain, final GPU acquire
and compute join. Actual selected mutable ybuf/ss loads are coherent `.cg`, not
excluded `.nc`. Compiler proof `9f7d072` preserves resources and every explicit
fence/math operand; it drops only compiler-generated synchronization surrounding
the acq_rel atomic. No H200 cost exists. RMW, gamma representation and future
common inherited repairs must remain separate factors.

**Current hold:** root identified an inherited ATT7 ordinary-launch software
join progress gap in5/12 cells, present in G0 and607e. It is not an integration
delta or explanation for added cost. All private artifacts, including f042,
are held pending a minimal common attention repair/proof. Any later paired
experiment must apply that repair to both arms and use a new source epoch;
the present f042 source/host/protocol clearances are not whole-model readiness.

## Gamma consumer census and minimal alternative

Native `_fork_native_ma_prepare` at10156, bank assignment10173, still places
`weights['post_expert_norm'].contiguous()` in `ma_w[7]`. This is raw BF16 and
may alias the supplied tensor; **native preparation does not precompute** its
effective weight. Native host wrappers in both literals bind slot7 to
`const __nv_bfloat16* normw`.

The ledger enumerates all16gamma expressions per literal, complete balanced
expression, native line and enclosing device function. Only **four** helper
expressions lie in the selected short WS functions: two integer-bit-loaded
gamma values in `skinny_ring_ws`'s last-B tail, and two BF16/direct-loaded
values in `ws_helper`'s B epilogue. `run_phase` selects skinny_ring_ws under
WS1; Tiny has no fat jobs, and MID SPLIT1 bounds jobs to8rows. Other literal
sites in epi_A/B, old skinny_ring and helper WGS code are not evidence of16
executed short conversions. Function/statement counts are not dynamic counts;
gamma is already prefetched/amortized over some epilogue work.

Feasible alternative: prepare a new immutable8192-byte BF16 coefficient
`(raw.float()+1).bfloat16()`, pass it at native slot7, and replace all native
gamma-add/round expressions by direct conversion of that effective coefficient.
Direct BF16, cached BF16 and unsigned16-bit→float load forms must agree. Leave
GEMM2 scaling/BF16 rounding, norm/sum arithmetic and every publication edge
unchanged. Raw input weight storage and Marlin effective-gamma state must remain
independent. Current607's native fallback is Marlin, already preparing its
effective coefficient. The separate owned-adapter tool also prepares an
effective coefficient through its own ABI, but is not selected in this607file.

This removes the new effective-weight rounding **and the old FP32 add** that
G0 already performed. It is a coupled host/native format alternative, not an
isolated measurement of just the newly introduced rounding instruction. Its
cost and interaction with ILP/register allocation remain unmeasured; neither
source-site counts nor8192bytes predicts a speedup. Root separately authorized
one OFF source/CPU prototype only after this ledger publication, with exhaustive
finite-BF16 scalar equivalence and actual consumer/host-selection tests. No
implementation or compilation of that alternative is part of this audit.

## Measured evidence and stopping rules

No per-repair paired H200 cost ladder exists. G0/607e headline ratio1.057715468
and short whole-load insertion factor1.043244918 are unmatched scenarios with
different routed banks and SGLang baselines. Do not add profile-row scenarios
or label them attainable repair savings. Main-only fixed-owner frontier
`1c6ebc76b91d4b7f` max1.004271 cannot explain short helper cost; it is not a
universal distributed-finalization bound. SC6/byte `05e51ddeb7014dd9` did not
clear repeatability/regression. Stage `63f5a2a8ed2341d1` measured
1.018493–1.028699 and is already recovered. Detailed observer f147 failed its
distortion gate; owned band32 a5c7/cached903e remain closed, unrelated.

The next permitted work is the small OFF gamma-format proof, not another broad
harness. Stop if a selected gamma consumer is unaccounted for, fallback/raw
storage is changed, coefficient math differs for any finite BF16 scalar, or
host/module format selection can mismatch. Future runtime attribution requires
same inputs/runner, conforming matched arms, common inherited progress repairs,
all four crosses/all three banks>=1% whole-six cold-graph ratio and>2xnull,
context regression<=max(.5us,1%), finite/reference/fresh/cache/generation/
capture/resource/budget coverage. No official/GPU action is permitted while the
campaign remains paused; root alone owns any later decision.
