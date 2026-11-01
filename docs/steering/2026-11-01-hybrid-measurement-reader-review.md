# One-arm hybrid measurement reader review — 2026-11-01

Reviewed root main `ff4cf42`, `tools/hybrid_layer_measurement.py` and its three tests. Read-only review; no H200 action. Existing tests pass locally. No scoring arithmetic error found. Follow-up fixes belong to root; this records findings at the reviewed revision, not a claim that later revisions still have them.

The reader correctly reconstructs each load from46 sliding and16 full weighted medians, then the six-load geometric mean. Weighted microsecond averages divide the weighted nanosecond sum by62000. Row outputs distinguish raw/overhead-adjusted medians and separately report actual per-job baseline/overhead. It explicitly labels archived/new observations unpaired and never returns a release decision. Score ratios across jobs remain confounded by inputs, routing, tuning, machine and baseline calibration; they are not isolated algorithm gains. No stale per-cell offsets are substituted into the new official score.

Owning-receipt safeguards to tighten before terminal interpretation:

1. `read` checks ID, board, public mode, tests/benchmarks/profiles and flags, but does not require top-level status=done and result.status=ok. Require both, in addition to current success fields.
2. CLI accepts a separate arbitrary startup log without binding it to receipt.output.log or a recorded own-job fetch/hash attestation. Wrong-job startup text could therefore qualify an otherwise clean receipt. Archived log_tail is only2325B of profiler tail and often omits startup, so tail-only validation is insufficient. Normalize the documented WSL/Windows output locator or retain a fetch attestation binding job/source/log digest; do not silently trust unrelated log input. No cryptographic log hash exists in the inspected raw receipt itself.
3. Resource parser checks generic positivity/residency/local0 but not all frozen startup bounds/per-kernel geometry. Its synthetic fixture intentionally uses576 threads for all three kernels and is accepted. Validate actual selected-kernel identities/threads/shared caps against the reviewed source/runtime contract, or explicitly keep parsing separate from resource qualification. Do not infer H200 values from that fixture.

Source content_hash/payload_ref should also be reconciled with the actual authorized upload manifest before interpreting the forthcoming job. The reader currently takes owning job ID but not expected source pin. Physical source SHA and normalized payload/content identity need their documented distinction; do not compare unrelated hash conventions blindly.

Optional reporting improvement: include weighted raw load averages/raw reconstructed sensitivity alongside adjusted load summaries. Existing per-cell raw values are already correctly labeled; this is additional visibility, not a formula correction or promotion criterion.

Pending terminal gate: root supplies exact owning job/receipt/log/source association; independently verify all13 tests,12 unique benchmarks/profiles,46:16 weights, startup selected4/checks48/captures8/fresh24/empty40/emptycaptures8/restored32, native resource/admission vectors, flags, preparation budget and score reconstruction. Keep failed receipts too. No terminal result exists in this review and no speedup is inferred.


## Same-day fix review

Root addressed the review in the working tree: terminal top-level done/result ok checks; server content_hash format plus payload_ref coherence; CLI log path must resolve to receipt.output.log and contain the job ID, with log SHA recorded; shared cap and MoE576-thread guards. Independently reran all four updated CPU tests: PASS. Reviewed reader SHA256 `bfb85988cd27358905cf08f85329f4adbf31dd60635e4902d7c79d4538dd1281`, tests `e2c5611c5ee6a7e3836824265cdf5c314250fdaa1a5ba564789bef2e87552dfb`. This fixes the identified terminal/log binding gaps for execution in the environment owning the CLI path (WSL). Exact ONE geometry remains supplied by its actual Instance getter rather than a guessed hard-coded value. Server content identity is still distinct from physical file SHA; root must retain the upload manifest association. No new official result is inferred.
