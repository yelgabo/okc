# Steering folder compaction, November 2, 2026

Start with the [shared owner queue](README.md). Decode32 has a [watched inbox](decode32/README.md) and [job ledger](decode32/STATUS.md). Together these three files went from 25,442 to 8,284 bytes against `bff7e503`, a 67% reduction in required reading by file size.

| Area | Reports archived | Supporting files archived | Where to look |
|---|---:|---:|---|
| Shared steering | 54 | 58 | [Shared archive](../../archive/docs/steering/) |
| Decode32 | 14 | 45 | [Decode32 history by experiment](../../archive/docs/steering/decode32/README.md) |

The [move manifest](../../archive/manifest-2026-11-02-steering.json) maps all 171 old paths to their destinations and gives the original and archived hashes. All 103 supporting files retain their bytes. Forty-five reports needed relative link changes. To find an older filename, search the manifest or run `git ls-files archive/docs/steering`. To reproduce a historical checkout, use the manifest's base revision.

Both queues retain their owners and next actions. In particular, General picked up GOPT-WS-03. Decode32 registered job `bd5776bdfcab42a7` for source `93a05524` and picked up the full-context rescale work. Its existing waiter owns that check. Read the [General pickup](2026-11-02-general-ws-norm-aggregation.md) and [Decode32 registration](decode32/2026-11-02-18-active-reduction-experiment.md) for the recorded decisions.

We kept active tasks, policies, paths read by tools and hash-pinned inputs in place. The original attention experiment, implementation plan and status still satisfy the D32-ATTN-01 manifest at their original paths. The MID result remains beside the record that cites it in the CUDA research catalog. The compact ledger links to the original receipts for completed experiments.

Verification covered relocation hashes, local citation paths, the current owner links, the D32-ATTN-01 manifest and the prior archive's 1,229 file hashes. Ten CPU tests passed across split-copy selection, B4 prefix selection, attention-pair source identity and the D32 comparison contract. The CUDA research checker still reports its pre-existing stale hash for `docs/ruled-out.md`. Its owning research review must reconcile that record before changing the catalog hash.
