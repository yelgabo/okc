# GOPT-SHORT-02: proposed helper scan is inactive

November 2, 2026, America/Vancouver. Pickup and source decision by OKC General Optimizer, chat01a18696, after reading the new task at evidence `ef50beab`. The user-authorized bounded task is complete by its source-closure condition. No new kernel or GPU job was created.

## Decision

**Close the proposed serial-to-parallel WGS prefix change for Tiny/Tiny2/MID.** The selected helper literal contains a serial `wgs_ranges` implementation, but the selected compiler flags preprocess that implementation and its x-producer out. None of T1/T8/T16 executes the proposed job-cost scan. This does not establish a ceiling for their active routing or other prologue work.

The complete incumbent remains `ba9e327c`, helper `b8ed8543`, B4/MAIN native `c7c201ae`. Actual final-factory requests use `OKC_WS=1` for all three helper modules. They do not override the helper's `OKC_WGS=0` default. The serial WGS region, including `wgs_prefix`, `wgs_range`, `wgs_ranges` and `wgs_xproducer`, is guarded by `#if OKC_WGS`. The active extra warp instead runs `ws_helper`. Compute thread0 publishes four `S.wsr` boundaries from `S.njS`, using `muldiv_g`, before `WS_JOIN`. That block does not scan a job-cost prefix. Tiny/Tiny2/MID already use eight-row skinny jobs; this result does not propose changing that partition or their ring.

[Original task](2026-11-02-general-short-prefix-task.md), [own audit tool](../../tools/gopt_short_prefix_audit.py), [complete selected-source and preprocessing record](evidence/2026-11-02-gopt-short-prefix/audit.json). The record retains exact flags, source hashes, selected tokens, module identities and active kernel bodies. B4/MAIN retain their existing parallel WGS implementation and were not edited.

## Validation and limits

`python3 tools/gopt_short_prefix_audit.py --out docs/steering/evidence/2026-11-02-gopt-short-prefix` passed all three real helper builds using CUDA12.8.1 `nvcc -E` on the device prefix with their actual factory flags. All three preprocessed outputs exclude the WGS functions and preserve `ws_helper` plus the four range writes. Compressed preprocessed outputs and full preprocessing logs are adjacent to the JSON.

A synthetic first-helper control adds `-DOKC_WGS=1`; the same checker then observes the serial WGS prefix and x-producer. This control checks detection of the preprocessor gate. It is not a valid new kernel configuration, performance candidate or GPU run. The host Torch extension tail is excluded from preprocessing. No compiler optimization, numerical equivalence, runtime correctness or latency gain is claimed.

The initial parser expected unexpanded CUDA qualifiers; preprocessing expands them to attributes. The checked parser locates the stable `moe_fused(Params p)` signature instead. No source or compiler failure was filtered into a passing result.

The prior D32 source audit correctly identified which literal each factory selects, but its helper conclusion inspected the raw function text without preprocessing its feature guards. That is insufficient to establish execution. The root instructions now explicitly require selected-flag preprocessing for such assignments. Original task/history is preserved with this dated correction.

Searched current run and ruled-out ledgers for the proposed helper scan treatment. No duplicate measurement was needed: exact preprocessing already makes this treatment a no-op. NS4 remains separately parked under GOPT-PIPE-01, own jobf0b875484960438b. No completed job was polled again, no active owner's job or files were changed, and no benchmark was launched.

## Next queue

General Optimizer needs a different source-qualified task. CUDA Optimizer keeps the separately assigned B4/MAIN phase-attribution study; Decode32 Worker keeps its accepted strict routing diagnostic. This closure is not a transfer of either task.

The standing goal's refill request is [posted separately](2026-11-02-general-hotspot-search-refill.md). It asks the named public-code scout and CUDA research owners for mechanisms on the actual compiled WS helper or a different nonoverlapping incumbent bottleneck. Posting in the repository is not verified direct-chat delivery or pickup. There is no new GPU allocation, register/ring sweep or blind port of the inactive WGS scan.
