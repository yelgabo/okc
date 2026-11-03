# Historical standalone MoE best: progress-contract qualification gap

**Confirmed on the exact historical source and owning receipt:** public MoE best `ec565a7` has the same ordinary-launch/software-grid progress and nonfatal-timeout gap recently repaired in the layer copy. Both selected kernel families are affected, covering all six scored loads. Preserve its historical result; this audit does not withdraw, change or submit anything and supplies no observed new GPU failure.

## Exact identity and actual paths

`git show ec565a7:muse_moe_w4a16/submission.py` is235700 bytes, SHA256 `b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14`. Owning retained receipt `muse_moe_w4a16/profiles/scored.abe7651dbaa44458.json` is board `muse_moe_w4a16`, id `abe7651dbaa44458`, done/pass5, checks.ok, no flags, score **2.2157115620135492** with95%CI **[2.182475888583346,2.2489472354437523]**. Its upload content hash is `084911f39cdd77d1d9ea6130d42b01cfe3a928b3fff3f8ac375ada4b444984ea`, also present in owning check `b50351288b504e81`. The retained runs log links that check/source/submit; upload content hash and raw Git file SHA are different identifiers, not claimed equal.

The receipt's actual `POLICY` line and all six GPU profiles agree:

| Scored load / tokens | Actual selected kernel | Launch |
|---|---|---|
| decode1 /1 | sk_8_3_0::moe_fused_kernel<1> | ordinary |
| verify1 /8 | sk_8_3_0::moe_fused_kernel<1> | ordinary |
| decode16 /16 | sk_8_3_0::moe_fused_kernel<1> | ordinary |
| decode32 /32 | sk_8_3_0::moe_fused_kernel<1> | ordinary |
| verify16 /128 | moe_main_kernel<47> | ordinary |
| verify32 /256 | moe_main_kernel<47> | ordinary |

`launch_main<MODE>` calls `moe_main_kernel<MODE><<<sms,kNTM,sizeof(SmemMain),stream>>>`; skinny `launch<SPLIT>` calls its corresponding ordinary `<<<sms,kNTL,sizeof(Smem),stream>>>`. They set the dynamic shared-memory attribute but make no cooperative request or whole-grid occupancy admission check. Legacy occupancy/cluster/PDL helpers elsewhere in the file are **not these selected launchers**. The source's generated dispatch maps the recorded variants to these functions.

## What is unproved

Both main and skinny contain generation-alternating software grid joins at `ctr[4+(gen&1)]`: each CTA publishes then polls for all G arrivals before cleanup. Other readiness waits also exist. Common `spin_failed` records `ctr[8]` and returns true; callers break/return from the poll and continue, rather than aborting before dependent reads or cleanup. Historical policy self-tests printed spin flags0 and official outputs passed; that is empirical evidence on those runs, not a general scheduling/progress proof.

CUDA12.8 §2.2 permits arbitrary, including serial, block scheduling. §8.7 provides cooperative launch plus resident-grid capacity calculation for documented grid co-residency. An sms-sized ordinary grid and shared-memory upper bound do not request that guarantee. This is an inference from the source and contract, not proof that a historical test deadlocked. [NVIDIA CUDA12.8 Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#grid-synchronization-cg). A fatal timeout can instead use the abort semantics of [PTX8.7 trap](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#miscellaneous-instructions-trap), retaining publication fences and preventing normal fall-through.

## Recommendation and next gate

Treat abe765 as the **historical measured best**, not a newly proof-qualified standalone release recommendation. No qualified replacement of these exact bytes has been established here. Preserve its score/receipt and production bytes. No slot, withdrawal or new dispatch is authorized by this audit.

Existing reusable building blocks are `moe_owned_cooperative_pair.py` (two ordinary standalone wrapper replacements; checked cooperative control was exercised in835253441a214a1c) and `verify_owned_progress.py` (fatal shared watchdog plus a separately qualified layer launcher). The835 common control is a later/different source and parameter policy, not a repaired ec565 release. Neither source-pinned transformer may simply be applied outside its accepted pins. A future delegated pure repair must cover **both** recorded wrappers, preserve legitimate local/chunk interfaces and all math/fences, reject unsupported admission, trap every active polling timeout, and retain strict full-six-load correctness/fresh/empty/reuse/resource gates. Any precision/parameter changes require separate explicit attribution; no projected speedup follows.

[Read-only evidence/reproducer output](evidence/2026-11-01-historical-moe-progress/audit.json) includes source/helper pins, exact receipt policy, source excerpts and each owning profile. Reproduce with `python tools/moe_historical_progress_audit.py <root> <output.json>`; it uses Git and retained receipts only. No GPU or production mutation occurs.

## Read-only reuse plan requested after the audit

No implementation was performed. Source inspection confirms both historical wrapper signatures and their existing `Params`, `SmemMain`/`Smem`, `kNTM`/`kNTL` match the standalone `moe_owned_cooperative_pair.py::MAIN/SKINNY` insertion sites. Its source guard intentionally rejects ec565: it accepts the later multiply-control/treatment pins only. Its `p.lo==0 && p.nloc==32` gate is also narrower than the historical `_run` API, which forwards `config.local_start/local_count`. Do not bypass that guard or silently restrict legitimate sharding. The layer `LAUNCH` additionally references `p.pdl`; historical standalone `Params` has no PDL member, so it is **not** a drop-in replacement. Reuse the checked CUDA query/admission pattern and namespace-specific two-wrapper instantiations, with a new exact ec565 source guard and separately reviewed compatible local/chunk bounds.

One common historical `spin_failed` definition is visible to both main and generated skinny namespaces. Its exact diagnostic-atomic/return anchor matches the fatal-trap pattern in `verify_owned_progress.py`; a future source-pinned transform can therefore repair both families' polling expiry with one shared helper delta. Historical active launchers have no PDL option to disable. Preserve the two wrappers' actual kernel pointer, threads, dynamic shared size, stream and by-value Params ABI, plus every compute/fence/barrier/packing byte. Give the extension a new epoch; verify real `_cuda_source` dispatch expansion and cache/module identity.

A bounded future delegated qualification should:

1. Prove exact reversal of only two launch wrappers/common timeout plus any explicitly chosen preparation policy change; compile both selected instances (`moe_main_kernel<47>` and `sk_8_3_0::moe_fused_kernel<1>`) and check actual register/local/shared/admission. CPU mocks exercise success and fatal CUDA/query/shape rejection; timeout branch cannot return unpublished data. Do not merely re-label an existing ordinary file cooperative.
2. Decide explicitly whether to preserve the old tuner or freeze its recorded six-load policy. The original prepare builds/tests/benchmarks additional main/skinny/legacy choices; changing launch policy can change its selected table. A claimed pure repair of the historical selected paths needs the actual table recorded/checked, not inferred. The old final `_self_test` returns a boolean which `prepare` prints as `ok` but does not itself assert at return; the new qualification must fail closed on its final selected-path checks.
3. Preserve strict FP32 reference global0.008/row0.075 checks (row failure must actually reject), finite reference/output, all six loads plus33/257 and the512-chunk boundary, real global256 IDs/local offsets, normal/skewed inputs, empty/all-remote and occupied↔empty transitions, generation reuse, scratch guards and long captured replays with changed input/ID/weight buffers. No manual counter resets between occupied/empty calls. Require exactly observed coverage rather than a successful partial prefix.
4. Only after source/CPU/local compiler review, root may authorize one official standalone free check with all5 official tests/all6 benchmarks and actual cooperative attributes/graph behavior, no flags and complete setup within300s. Future optimization comparisons should start from that qualified safe control. Historical abe765 and835 are archival comparisons, not simultaneous paired timings for the new exact source. No slot is authorized by this plan; first-two G reset intent/order remain unchanged.
