# SK-XOR owning-receipt validator, 2026-11-01

Source checkpoint follows root's frozen private harness `154ac09`; the exact upload artifact remains `e5d82cf2f1245495e070f6953a6f0b048cf21ac786b0cf4aa97384d3fc6f7020`. No H200 experiment, result, public action or quota change occurred here. All new test receipts are synthetic.

`tools/proj_sk_xor_receipt.py` is a separate post-run reader. It does not modify the private artifact, codec or immutable scoring policy. `proj_sk_xor_read.py` stays SHA256 `00e99c9e51423bf349f69ac3aed7f58838c0bce788c59292289d886055b2e694` after normalized text loading; the validator checks this against the frozen manifest before using it. No source text is compressed/decompressed: SX2 decoding covers measurement data only.

## Fail-closed attribution and coverage

The caller must provide `--job`; the terminal raw receipt must own that exact16-hex ID, board `muse_bf16_proj`, mode `check`, status `done`, H200, passed/ok/returncode0, no error/flags/check rejection, nine passing official test records and thirty distinct passing benchmark cells of weight62. Public `run` receipts and another job's result fail even if their headline looks plausible. Official per-cell positive finite raw/adjusted medians are required.

Exactly one SXPINS, SXREPORT1 and SXDONE line is required. The codec payload must remain below9000 characters and decode fully. Both printed and decoded pins match every committed harness-manifest field except physical source SHA and byte size (the current manifest has seven such fields, not eight). The prototype's raw/LF/runtime pins must agree. Source SHA is not confused with the judge's content hash.

The reader requires complete preparation before295s, the frozen three seeds/24 rounds/90-row schema, and exact integer coverage counters:12 immutable formats,210 placements,840 captured-output oracle checks,420 fresh checks,210 post-timing checks,80 one-hot checks and80 outlier checks. Canonical `score_model` validates every row/arm/vector, and recomputed raw output is deeply compared with the printed raw_gate/raw_pass and final footer. A genuine complete performance loser remains a complete result, not an exception or hidden raw record.

## Resource and ownership reconstruction

The ten resource groups must each contain c0/c1/t0/t1, expected independent module pair, native cfg/variant/non-nearby IDs from the committed prototype manifest, and correct actual weight/output extents. Static/dynamic shared memory, thread roles and maximum threads are recomputed independently from each committed `original_targs` plus reported pmax, then compared to both actual getter vectors and the two reported layout dictionaries. Expected dictionaries alone are never accepted as proof. Local memory must be zero; duplicate builds must match; candidate blocks/SM must be at least control. Register×thread products, immutable payload/header/table/row-map byte extents and duplicated immutable pack metadata are checked.

The actual C2 anchor ABI and base/FULL indices are pinned to the frozen3044ee7 configuration order. A separate [CPU source crossproof](evidence/2026-11-01-projection-sk-xor-receipt/anchor-source-crossproof.json) constructed all ten closures from the exact d23aa17… augmented source and matched those indices/free variables without CUDA execution. This confirms the reader's ID constants; it does not substitute for the future H200 prepare's actual anchor guard. Reported inventory must include all five exact BF16/FP32 weight shapes.

## Raw versus contemporaneous calibration

The primary raw model is recomputed from all vectors. The same owning receipt supplies all thirty offsets `(median_raw_ns-median_ns)/1000`; the identical offset for a cell is applied to every arm in a separate sensitivity model. No stale9eba offset is used. All positive adjusted domains are evaluated by the same unchanged pure policy. Registered decision is raw.keep AND adjusted.keep AND valid domain. These are private paired event factors/common-offset sensitivities, not an official paired candidate score.

If any adjusted latency is nonpositive, the reader returns `calibration_domain_valid=false`, `registered_decision=false`, the reason, and `raw_complete=true` with all raw vectors and raw gate preserved. This is an invalid calibration domain, not evidence that the algorithm failed its raw gate. The unchanged20 operators retain one actual common vector per cell/seed, identically charged; they provide no independent paired null.

## Validation and next action

Eleven synthetic CPU tests pass ([log](evidence/2026-11-01-projection-sk-xor-receipt/cpu-tests.log)): valid contemporary reconstruction; invalid-domain preservation; positive-domain adjusted loss; mandatory CLI job; wrong job/board/public mode/nonterminal result; flags/test/benchmark failures; missing/truncated/duplicate log messages; malformed rows/counts/pins; resource/native/ownership/layout/header failures; false printed gates; and complete genuine loss handling. Header/source provenance uses the committed manifests, not ignored scratch files. In the isolated worker, `PYTHONPATH` and `SK_XOR_EVIDENCE_ROOT` pointed explicitly to root's current immutable policy/evidence; in the integrated tree normal sibling paths resolve them.

Future invocation, after root reviews and an authorized projection check actually finishes:

```powershell
python tools/proj_sk_xor_receipt.py RECEIPT.json --job EXACT_OWNING_JOB --output ANALYSIS.json
```

Stop on metadata/source/coverage/attribution/resource disagreement. Preserve an invalid adjusted domain as specified. No GPU/readiness or speed claim follows from these CPU tests; root still owns review, dispatch and any subsequent exact pure qualification.
