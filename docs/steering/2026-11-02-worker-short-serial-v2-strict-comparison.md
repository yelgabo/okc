# Serial terminal archive: strict offline receipt comparison

November 2, 2026. Follow-up to captain fresh-profile checkpoint **2a59899592d57a2f90cee0e0944b75c1c8e4fe93** and existing worker terminal publication **87cfa4afb1328d51de4924570adc37ff1f3cca5a**. Allocation remains **1df00005d3eceb467c38b17ac0d4cad632efbc78**, already consumed by the two completed ordinary private checks. No new service call, GPU job, timer change, retune or qualification cycle.

Executed the existing tools/dl_compare_receipts.py OFFLINE against the immutable complete control/terminal-status.json and candidate/terminal-status.json, with explicit control job **cc474883117947c0** and candidate job **e4c8f00532b74d15**, default check modes. Exit 0. Raw Python JSON stdout is saved directly, without JavaScript JSON reserialization of 64-bit seeds. Exact command, timestamps and SHA256/length pins for the comparator and both input receipts are retained in strict-receipt-comparison-provenance.json.

## Validation and decision

The strict utility accepts both owning layer/check/yelgabo identities, completed successful receipts, server content/payload association, 13 tests / 12 benchmarks / 12 profiles, case/weight associations and complete 46:16 score reconstruction. This does not independently download server payload bytes or replace physical-source publication pins.

- Adjusted own-latency factor: **0.9698272866341408**.
- Candidate-reference/control-reference baseline factor: **1.0232267932435573**.
- Their product equals official candidate/control score factor **0.992353264502752**.
- Affected short factor: **0.9625514695959178**; unchanged-long factor: **0.9734857931467625**.
- Descriptive short-relative-to-long whole-score scenario: **0.9962418490969489**. Not a randomized causal estimate, new retroactive gate or confidence interval.
- descriptive_screen=false, performance_qualified=false, promotion_authorized=false. **Exact treatment remains PARK/STOP; no reserved confirmation request.**

Selected paths were already archived in the [terminal report](2026-11-02-worker-short-serial-v2-terminal.md): all four short candidate cells have separate original pooler plus FP32 router and no FRT; control has FRT and no separate FP32 router. All eight long symbolic kernel-name/callcount lists match. These are path observations, not exact routing/cache/scheduling equality. Separate sessions and all twelve differing benchmark seeds retain the causal limitations even on the same reported node.

## Evidence and ownership

[Raw comparison](evidence/2026-11-02-worker-short-serial-v2/strict-receipt-comparison.json), [execution/input pins](evidence/2026-11-02-worker-short-serial-v2/strict-receipt-comparison-provenance.json), and retained empty stderr sit alongside, not in place of, the immutable terminal manifest. A separate strict-comparison-manifest.json pins this addendum and new artifacts. Prior registration, source, raw profiles and reports are untouched.

Captain owns README/shared-ledger reconciliation. One private check remains reserved per the separately archived 13:03 UTC read-only snapshot; this offline execution does not consume quota. No public/scored/default action or top-100 claim. Further measurement requires a new separately justified allocation, not replaying a delayed instruction.
