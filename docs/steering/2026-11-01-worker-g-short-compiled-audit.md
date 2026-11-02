# Family G shorts: exact selected compiled audit after RMW null

November 1 Pacific / November 2 UTC 2026. Evidence base5fdbd563114b0af7066a4727e0f7d1dbcf9aa9d2. Bounded source/CPU/device-compiler audit only; no subagents, GPU, judge/CLI dispatch, production edit or new performance candidate. Captain owns raw-receipt/profile reconciliation and README; researcher owns arithmetic/active-policy proof.

## New finding: not short-kernel resource inflation

Exact common OFF4cbb13a9 and public607/ba9 select **byte-identical Tiny/Tiny2 CUDA, CPP and complete factory flags**. This is not the paired instrumented ON or a dormant alternate literal. Selected helper SHA b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716, device prefix80ea6081901508d325d36650c2d6a3b739bc6d8f2649f804e12bc5de36874efe. Their common-control repairs therefore introduce no new short native instruction stream in these captures. Whole-layer preparation/other components remain outside this identity claim.

Donor source pin is upstream8e30e1be994db4310794b8d04e90d18d895deb84, raw1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a, LF82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19. Actual short helper is84f501957a280cb3703bf9a29843153557e280725457162cb8b203b74b8d0f9f, not its MAIN literal. Public607 physical sourceba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab; OFF4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280. The native factory's actual flags match donor versus repaired per family. Tiny uses WSBLK2; Tiny2 uses WSBLK1/CPF0. Neither selects the standalone donor's dedicated M8 ring issuer.

Existing public/common matched12.8 artifacts were reusable; scoped .build and steering manifest inventory found no exact donor short artifact. Consequently only four fixed builds were made: donor/public607 Tiny/Tiny2, both bool entries. Common OFF reuses public output by full source/flag identity. Existing NVRTC13/PTXAS13 device-only adapter, no installation or sweep. Actual M1/M8 selected entry is false; both entries give the same rows below.

|Secondary sm90a selected entry|Donor|Public607 = common OFF|
|---|---:|---:|
|Tiny registers / barriers / stack / spill stores / spill loads|147 /3 /0 /0 /0|same|
|Tiny2 registers / barriers / stack / spill stores / spill loads|147 /3 /0 /0 /0|same|
|maxntid / minnctapersm, both families|288 /1|same|
|Tiny static instructions / non-NOP|5232 /5194|5272 /5234|
|Tiny2 static instructions / non-NOP|5272 /5241|5320 /5283|

No secondary compiler register/spill/stack/bounds regression is found. These are not target NVCC13 attributes, measured occupancy or dynamic memory transactions. Target donor CUDA resource readback is unavailable. Different12.8 instruction totals in older reports are not replaced with these13 counts.

## Remaining compiled dependencies and protocol edges

Both families add16 scalar F2F.BF16.F32 instructions, two FENCE.VIEW.ASYNC.S sites, and associated widening/address/scheduling changes. Counts include every emitted selected-entry path, not dynamic executions. The new gate chain is scaled gate FMUL -> scalar BF16 round -> integer widening -> log2 multiply -> EX2; donor lacks the round/widen dependency. Up remains independently rounded later, rather than making EX2 wait for an artificial packed gate/up pair. GEMM2 scaling-before-round and effective-gamma rounds also contribute. These retained boundaries cannot be disabled as a timing arm.

The two extra proxy fences are **two emitted phase-A/phase-B copies of one skinny_ring_ws source bridge**, not two unrelated new synchronization mechanisms. Each unit generically reads its ring slot, finishes register MMA/reduction, waits for the partial buffer, stores S.red, performs the per-reader proxy fence and CTA_SYNC, then thread0 may issue the asynchronous refill into the recycled slot. The phase-A and phase-B retire edges are distinct dynamic uses of this same lifetime obligation. The bridge is not merely an extra fence publishing S.red: its placement also orders the preceding generic ring reads before async reuse. Existing source repair contract requires every reader, not only thread0, to carry that edge. No cheaper proven representation of this bridge is identified; its chosen late placement after the partial-buffer wait/store is not proved uniquely necessary. No fence was removed or moved here.

Separately, final-counter strengthening adds an acq_rel returning RMW instead of donor atomicAdd. Predicate-aware counts identify another CTA membar and GPU membar/cache/error-ordering sequence from this change; do not conflate those with the two reader-retire sites. The valid c5e4f89672f743bb matched RMW result failed the registered1%/2xnull promotion gate; it does not erase the arithmetic/reader-retire differences, reject every atomic design, or establish causal T16 regression. No#4 rescue.

Donor host source supports ordinary/PDL launch alternatives; repaired source uses checked cooperative admission and a same-stream launch with per-call resource/bounds checks. Resident-grid progress is required; this particular API/policy is a conservative implementation choice. Host wrappers were not compiled/timed, and host API calls made while recording a CUDA graph cannot simply be charged per timed replay. No concrete per-edge PDL/admission proof or cheaper qualified alternative is supplied by this audit; the researcher's policy proof remains separate.

## What this does not explain, and exact next tests

The40/42 extra non-NOP instructions are not40/42 measured delay cycles or a causal whole-score tax. Captain's receipt reconciliation finds different H200 nodes and all12 seeds different; no causal5.77% claim is supportable. The compiled evidence narrows plausible mechanisms to retained arithmetic/dependencies and protocol/launch policy, without assigning latency to them.

Respect prior negatives: [exact packed-RNE](2026-11-01-g-short-pair-rne.md), source SHA256 `19cfb4eeeb0dde21b142a23a880e8f921dfd02a63806a05649c40e6cc54e6a16`, replaced12 scalar converts by6 packed but added8PRMT and a gate-on-up dependency; it is not a scalar-scheduling solution. [Gamma12.8](2026-11-01-legal-frontier-research.md) reduced static instructions but failed MAIN resource preservation; [gamma13](2026-11-01-gamma13-main-screen.md), published241c04a, independently worsened MAIN stack/spills. Short gamma resource parity alone was never a measured gain. None is reopened by this audit.

1. **Exact-operator scalar scheduling test, offline first:** keep each scale, scalar BF16 round/widen, divide, up multiply and final BF16 store exactly ordered; keep gate independent of up. Test one deliberate interleaving that makes independent up/down preparation overlap the rounded-gate EX2/divide chain, never the19cf packed form. Compare selected Tiny/Tiny2 PTX/SASS for an actually shorter gate chain or useful scheduling slack with identical numerical operators, fences, launch bounds and no worse resources. If emission is unchanged, adds gate-on-up dependency, or worsens resources, stop before runtime. A changed static schedule still needs separate correctness and paired full-six permission; no speed forecast or implementation allocation is implied.
2. **Per-edge PDL/admission test only after proof:** require exact producer publication, consumer wait, descriptor lifetime, resident-grid progress and capture/replay proof for a named predecessor->expert or expert->consumer edge. Only then compare identical safe arithmetic/reader-retire kernels under the proved alternative versus cooperative same-stream control. No ordinary oversubscribed donor rollback and no generic observer harness. With no proof, this test remains NOT READY and cannot consume a check.

## Evidence and compiler limits

[Source/factory capture](evidence/2026-11-01-worker-g-short-compiled-audit/source-audit.json), [four-build manifests and hashes](evidence/2026-11-01-worker-g-short-compiled-audit/compile13.json), [predicate-aware opcode deltas and exact dependency windows](evidence/2026-11-01-worker-g-short-compiled-audit/instruction-audit.json), and per-build compiled/ folders preserve compressed PTX/SASS and raw compiler logs/manifests. The donor-to607 source patch is retained; public-toOFF patches are empty. Actual helper names are tools/g_native_repair_inventory.py and tools/g_native_patch_ladder.py; existing pure-binding/factory methods are reused, not submission imports.

Compiler scope: existing secondary NVRTC13/PTXAS13, compute_90a/sm90a, captured defines, C++17, lineinfo; existing virtual stdint/type_traits and declaration-only wgs_cta_of adapter; device prefix stops before Torch wrappers. Actual NVCC13 host-preprocessor absence remains the previously documented limitation, not retested or repaired. Initial NVRTC library search failed to locate already-installed builtins; setting LD_LIBRARY_PATH to the existing cu13/lib allowed the fixed four builds, with no install or source treatment. No CUBIN executed. Exact adapter and artifact hashes are in the receipts; its reusable source is in the prior gamma13 compiler-artifacts archive.
