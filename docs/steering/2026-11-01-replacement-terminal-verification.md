# Replacement worker: terminal archive independently verified

November 1 Pacific / November 2 UTC 2026. Evidence checkout main at 7076754; implementation d5f29167, acceptance cc80a2c. Sole source/CPU worker; no subagents, GPU dispatch, status polling, quota query or production edit.

## Verified outcome

Private c5e4f89672f743bb is the completed H200 receipt, not a new measurement. All 19 files in terminal-archive-manifest.json independently match exact byte counts and SHA256 hashes. Physical upload785a90ca/924005B, manifestc5f0e397/4821B and unchanged reader745d7bb5/10920B also match. Local Python3.14.5 frozen-reader rerun exits0, complete=true, private_raw_gate=false. All nonfloat fields and the complete reported GPU capsule match exactly. Two derived host math sums differ by about2.2e-19 across platforms; both original and rerun values are retained in replacement-verification.json. No original archive file was rewritten; no gate threshold or frozen reader was changed.

Official13tests/12benchmarks/12profiles PASS with setup178.253237769s. The12 matched bank/cross whole-six logs remain -0.001829079..0.002828144, failing1% and2xnull; bank993821 additionally fails its context guard. Official repaired-common OFF2.112499365 CI[2.080811875,2.144186856] is descriptive, not an ON promotion result.

## Decision and handoff

RMW-only promotion CLOSED; no conditional#4, seed/mask rescue or remeasurement. Existing wait11629 is terminal and was not polled/restarted. The full terminal report, run-log row, ruled-out row, registration addendum and exact19-file archive are published together with this verification. Captain owns README/index and must link the full terminal report and this verification; this worker deliberately does not edit that shared file.

Next allocated work is a bounded source/CPU screen of one universal raw-BF16 SiLU table, rules first. Eligibility conflict means STOP/HOLD before a full harness. No numerical GPU, timing, release or target-toolchain equivalence claim follows from this verification.

## Evidence and reproduction

- Full result: [terminal report](2026-11-01-rmw-legal-fixture-terminal.md).
- Exact19-file inventory: [manifest](evidence/2026-11-01-rmw-common-pair-v4/terminal-archive-manifest.json).
- Independent receipt and two platform-rounding deltas: [replacement verification](evidence/2026-11-01-rmw-common-pair-v4/replacement-verification.json).
- Reproduce locally: python tools/verify_rmw_terminal_archive.py. Requires the frozen .build/rmw-pair-v4/check.py; no GPU or network is used.

Gaps: no current-now standings or quota query; no new candidate, speed gain or proof of the old v3 device-trap cause. Top100 overall remains the objective, not an achieved result.

