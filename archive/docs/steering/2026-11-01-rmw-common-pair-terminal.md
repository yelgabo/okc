# Repaired-common paired check: terminal setup failure

November 1, 2026 Pacific / November 2 UTC. Exact implementation/preregistration **2686a7653d6cf2325e944f419cdf0f81e4b6308a**, accepted-job evidence **c17d4a5aa17441ab40e8091018eece9e2cf328ea**, queue checkpoint **0b4b7454da2522e037f0bf1468692f2564d939b7**. Own private tranche #2/4: **16b4db27421d4714**. This report supersedes queued snapshots for that ID. No new check or public submission is made.

## Decision

**SETUP FAIL; attribution unavailable.** The owning terminal receipt says done, result status ok, **passed=false**, judge return code **112**. Its only record is setup/fail with CUDA unspecified launch failure, driver error **719 / CUDA_ERROR_LAUNCH_FAILED**. Static checks pass with empty flagged/rejected lists; they do not establish execution success. The CLI wait exits **1**, consistently with failed evaluation despite result.status=ok and result.error=null.

The frozen owning reader rejects the receipt at its terminal official-success guard, before interpreting any capsule. There are **zero official correctness tests, zero benchmarks and zero profiles**, and no score or CI. No SRM_COMMON_REPORT capsule was emitted. Per-load 46:16 changes, aggregate log gain, four crosses, duplicate nulls, regression gates and heldout promotion are all **unavailable**. This is neither an RMW performance loss nor evidence that repaired common OFF or ON fails legal production workloads.

Stop this exact setup allocation for specific diagnosis. Do not retry, dispatch #3, change thresholds or infer public/scored permission. Two of four initial tranche slots have accepted IDs; the originally reserved fresh heldout #3 is not repurposed by this failure. No quota refresh/decrement/refund is inferred from the terminal receipt.

## Owning identity and timing

- Board muse_decode_layer, mode check, team yelgabo; own ID16b4db27421d4714.
- Physical upload .build/rmw-pair-v3/check.py: **923915B**, SHA256 **c4a5ee29bcfeca248c0fee638ed103e9f211f7a8cc32b86bd3bb642676bf45e3**; source reverified before reading.
- Server content hash **09017717fcca125fa947d2e8e143da7409080668cd55c77d6229bf252d6ebed1**, matching payload_ref. This is not the physical file digest.
- Frozen reader tools/g_common_rmw_pair_read.py: **745d7bb508d170dbc6bcd4b86e0f8a0b39c9a9ba08885aec54a45a33e68f004d**. Manifest **09f2132b25cb1ec8456efcf2ab0f9197780d4612f31bac710eaee27c5db14cf4** remains unchanged.
- Accepted **2026-11-02T01:31:40.555270Z**; worker started **01:56:05.829248Z**; finished **02:00:03.410670Z**. Queue **1465.273978s**; worker start-to-finish **237.581422s**. Worker wall time includes evaluation activity and is not prepare duration.
- H200, node n-nzf2vzn536ewibf7w3dgpswb6g, driver595.91.07. Exact evaluator/image/file and before/after machine metadata are preserved in the receipt. No successful resource-vector capsule is available.
- Setup prepare_s and prepare_pad are null. No phase or cleanup timings were emitted; neither 295s fit nor a preparation timeout is established. The observed failure is a CUDA launch failure, not a budget assertion.

## Observed failure boundary and limits

The owning log prints repaired native routing/backend selection and **G_RELEASE_STARTUP_OK native ... 56 verified_replays**. It then traces prepare -> _srp_prepare -> _wl_prepare -> _srp_occupied_empty -> **graph.replay(); torch.cuda.synchronize()**, generated submission line17841 / runtime support line80. It does not print WL_PREP or the final capsule.

Source order puts this inspection after mutable ownership/resource and empty checks and before the 36 matched rows. Reaching this source location is evidence of that control-flow boundary; it is not a preserved independent resource vector or a completed all-six correctness result. The trace lacks token count, arm ordinal, occupied/empty transition and native instruction. CUDA explicitly warns that asynchronous reporting may move the observed stack location. Do not attribute error719 to a particular trap, arm or transition from this log alone.

The newly added occupied fixture initializes IDs to0 in every one of8 slots and repeats that pattern for each token; later occupancy transitions fill the same tensor with0 or-1. Production routing selects distinct top-k expert indices. This is a concrete fixture-contract concern, and tiny source contains a fat-unit trap. Its exact multiplicity/predicate/selected-arm implications require the bounded SOURCE/CPU diagnosis requested by captain after this archive; the current receipt does not localize the device fault. No fixture or production source is changed in this terminal-handling commit.

The wrapper has a finally cleanup block, but a failed CUDA context can prevent CUDA cleanup calls from completing. No terminal cleanup-restored capsule exists, so successful cleanup/resources/actual-attention coverage must not be claimed.

## Archive and validation

Raw owning CLI JSON is preserved unchanged, including integer precision, in [own-terminal-receipt.json](../../../docs/steering/evidence/2026-11-01-rmw-common-pair/own-terminal-receipt.json). The CLI transport records exit1 and empty stderr. The original owning log was copied byte-for-byte with a receipt-path/source-digest attestation. Source and frozen reader hashes pass; the reader then exits1 at its expected official-success rejection and produces no completed observation.

- [Terminal summary](../../../docs/steering/evidence/2026-11-01-rmw-common-pair/terminal-summary.json)
- [Reader rejection stderr](../../../docs/steering/evidence/2026-11-01-rmw-common-pair/terminal-reader.stderr.txt) and [reader transport](../../../docs/steering/evidence/2026-11-01-rmw-common-pair/terminal-reader-transport.json)
- [Archive manifest](../../../docs/steering/evidence/2026-11-01-rmw-common-pair/terminal-archive-manifest.json), containing byte counts and SHA256 digests
- [Canonical receipt](../../../muse_decode_layer/profiles/g-common-rmw-pair.16b4db27421d4714.json), [owning log](../../../muse_decode_layer/profiles/g-common-rmw-pair.16b4db27421d4714.log), [receipt log tail](../../../muse_decode_layer/profiles/g-common-rmw-pair.16b4db27421d4714.txt)
- [Frozen registration and stopping rules](2026-11-01-rmw-common-pair-registration.md)

## Next gate and ownership

Implementation owner retains dispatch ownership, but no next GPU action is authorized. After this archive, the bounded next task is SOURCE/CPU only: derive routing multiplicity, exact selected Tiny/Tiny2 fat predicate and legal top-k capacity; identify the contract violation; specify the smallest legal occupied-to-empty fixture and a CPU regression that distinguishes this hypothesis. Preserve failing pins and do not claim trap causality absent localization. Return the evidence and effort estimate to captain for an explicit narrow-repair/reallocation-versus-stop decision. No general harness refactor, new candidate, GPU run or production change belongs to that diagnosis. Captain owns README/index and receives this report path/commit for publication there.
