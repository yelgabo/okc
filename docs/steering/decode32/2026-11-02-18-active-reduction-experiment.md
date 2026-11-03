# D32-REDUCE-01: one accumulator copy per attention split

November 2, 2026 Pacific. The user directs continuous Decode32 work toward verified2.3x or better. An empty research queue is not a stop condition. Preserve other owners and closed variants, but qualify and execute bounded work from selected source and retained evidence while research continues.

## Objective and choice

The supplied historical SGLang C32 baseline22.149ms gives a2.3x planning target of9.630ms. The first lookahead candidate reported11.882275ms on different inputs and remains unconfirmed. Actual target qualification must use the owning run's baseline, held-out matched evidence and all-load gates, not substitute the historical constant.

Ran `tools/hotspots.py` on own17383751c6a24010 with the owning SGLang profile7d42771c3384449d. Its D32 sliding expert activity is73.3us and full attention172.4us. These are retained profile activities, not additive exposed savings. CUDA Optimizer owns the separate B4 phase study. This worker selects a nonoverlapping attention reduction experiment.

Three approaches were considered. Continuing the failed whole-cache observer repeats a closed instrument and supplies no candidate. A new two-kernel deterministic attention reduction would add a launch and more lifetime/ordering work. The selected smaller alternative uses the incumbent's existing multiple-copy merge, with one copy per split. It removes shared atomic destinations within this attention reduction and changes no CUDA code or build flags.

Retained attention evidence already documents FP32 arrival-order variation: job2b93c6ecf4f641c6 found deterministic slot-order merging removed repeat/history differences, with clean workspace scans. This is relevant prior evidence, not proof of the cause of D32's later route swap. The diagnostic mechanism used there is not imported. This experiment uses the selected G implementation's existing workspace API and finalizer.

## Exact candidate

[Generator](../../../tools/d32_split_copies.py), [tests](../../../tools/d32_split_copies_test.py). Active generated output is ignored `.build/d32-split-copies/candidate.py`; the exact source is preserved in [candidate.py.gz](evidence/D32-REDUCE-01/candidate.py.gz). Decompressed SHA256 `93a05524b1c10b44ac5bd1ea307a25fdc906db4a617737a579cfae6332fd0d49`,869598bytes. [Manifest](evidence/D32-REDUCE-01/candidate.py.manifest.json).

The complete frozen public607 prefix `ba9e327c` is unchanged. The only appended change wraps `_plan7`. For batch17..32/q_len1/sliding/page1/one KV pair/noncluster plans it sets `nc=n_splits`. Standard D32 sliding has8splits, so nc changes1→8. Full context and all other standard loads retain their plan. Explicit caller overrides remain authoritative. This candidate does not include lookahead or modify MoE.

## Mechanism, correctness and cost

In the selected per-request L1 path, copy is split index modulo nc. With nc equal to the split count, each split has a distinct accumulator row. The existing half merge leaves only wsub0 writing each head; lane/component addresses are disjoint. Each denominator has one writer per split/head. The existing eight-copy finalizer sums in copy order, applies the same normalization/gate/BF16 boundary and resets every copy. Original counters, joins, fences, progress and cache updates remain.

This changes FP32 association to a fixed legal ordering of split partials. It does not drop a numerical boundary or promise bitwise equality with arbitrary arrival orders. It can make this attention reduction reproducible, but other layer stages may still vary. Official math/cache/replay checks and the existing exact routing requirements remain.

The standard row uses4096workspace rows instead of512, adding931840bytes across accumulator/denominator storage, about0.89MiB. No new kernel launch or device compilation is introduced. The trade is fewer contended atomic destinations versus more finalizer loads/stores and cache traffic. The existing finalizer's different runtime branch may lose. This is one structural reduction policy, not a sweep of copy counts.

Three CPU tests pass. They execute the actual layer launch path across24 metadata cases and48 control/treatment launches, verify standard and neighboring selection/private workspace sizes, preserve explicit overrides, check the frozen prefix/hash, and enumerate unique split/head/dimension writers. [Complete output](evidence/D32-REDUCE-01/cpu-tests.txt). These are source/host/address checks, not target numerical or latency evidence.

## Preregistered measurement and decision

Publish the exact compressed source and verify decompressed remote bytes against the active file. Perform fresh shared jobs/quota preflight, submit one ordinary private full-layer check and retain its accepted ID before further measurement. Live collector output goes under ignored `.build/`, then is archived after exit. Verify the D32_SPLIT_COPIES marker shows32/1/2048/8/8, complete official tests and all12cells/profiles.

Report raw and adjusted D32 sliding/full, C32=46*sliding+16*full, owning baseline and all-six results. Historical94ef/cc474/public607 are descriptive context only. A promising ordinary result requires fresh matched confirmation before causal improvement or integration. No whole-score1% threshold is imposed on a valid small D32 gain.

Stop this exact treatment on failed math/cache/progress or no useful execution signal. Do not sweep copy counts or reinterpret a valid loss. If a valid signal survives confirmation, integrate the scoped plan and continue toward2.3x. Otherwise choose a distinct mechanism from source/evidence while research progresses. Existing public permissions remain unchanged.

## Accepted check and next local work

Own private job `bd5776bdfcab42a7` is accepted for source `93a05524`, published in `0585bd21`. Fresh shared preflight found no live jobs and15private layer checks. The single foreground waiter is exec session `47403`, writing ignored `.build/d32-split-copies/status-wait.raw.json`. Archive after exit and follow this own ID only.

Concurrent research published a distinct [full-context identity-rescale mechanism](../2026-11-02-general-hotspot-search-refill.md#2-decode32-exact-identity-rescale-elimination-in-full-decode-attention). Actual pickup is now recorded: Decode32 Worker will implement and compile one enabled/disabled pair while this accepted sliding check runs. No second GPU check follows until the current job is terminal and fresh shared preflight permits it. The new treatment skips only exact identity accumulator products and keeps denominator arithmetic and all synchronization. Its candidate resource gate remains open.
