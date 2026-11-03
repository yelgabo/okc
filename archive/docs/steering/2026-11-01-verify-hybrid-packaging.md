# Verify production packaging: implemented dispatcher, unresolved byte limit

November 1, 2026. This responds to the [08:02 coordinator assignment](2026-11-01-0802-g-progress.md).
Evidence starts at main `4c7233d`; integrated independent source reductions are
`75ead74` (original `4bee6b0`), `ec0bb05` (original `1ed36ee`),
`db39833` (original `e41cba1`) and final independent review `595aafb`
(original `5d245b1`). The dispatcher, complete matrix and19-test evidence are
pinned at implementation commit `0879082`. No GPU check or public submission
was dispatched in this task.

**The production dispatcher is implemented, but the nonzero overrides still
exceed the one-file limit. No new latency improvement or eligible submission
candidate is established.** The recommended submission remains the unchanged
607e incumbent. The goal of reaching the top 10–50 remains unfinished.

## Preserved evidence and actual scope

The incumbent is `muse_decode_layer/cand/g1b4_stage_release.py`, 869076 bytes,
SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
Its owning public receipt is `607eb52bdaf24160`: PASS13, 2.1260826764231093×,
95% CI [2.0941914362767626, 2.1579739165694556], preparation 127.759508348s.
Source: `muse_decode_layer/profiles/scored.607eb52bdaf24160.json`.
The coordinator's 08:02 snapshot records rank314 and layer rank50 point2.2894×;
these are timestamped observations, not a newly refreshed board or an overall
numerical cutoff. Packaging provides no new top-10 feasibility evidence.

The frozen private four-cell diagnostic remains 968389 bytes, SHA256
`2bad52ceca9ef72d73baa539dc9db0c776a51b6009a4e4a4f8a8d9b725761c0f`.
Its manifest is `muse_decode_layer/profiles/verify-four-cell.2026-11-01/manifest.json`.
It retains SGLang as its ordinary judged entry. It has no new H200 measurement;
do not mistake its sub1MB fit for a production hybrid fit or its restricted
two-load conditional sensitivities for a measured full-six-load score.

## Implemented production control

`tools/dl_verify_hybrid.py` and `dl_verify_hybrid_body.py` implement a fixed,
generation-time four-bit mask: verify16/sliding, verify16/full, verify32/sliding,
verify32/full. Each bit selects the complete corresponding owned pipeline.
All other cells delegate to G. Mask0 returns the exact original869076-byte
scored file, without a new wrapper or preparation path.

Nonzero masks use separate readable Python module literals and namespaces.
There is no source encoding/compression, runtime source-file read, timing-based
selector or branch on tensor values. The selector requires the fixed geometry,
page capacity, RoPE/config domain and raw-ABI dtype/layout/common-device
metadata; unsupported metadata delegates to G. Original routing, expert IDs,
precision boundaries and plans are supplied by the selected complete backend.
G startup validation is retained. The exact owned `_w4_scales_ok` predicate runs
on immutable weights during prepare; unsupported scales omit the owned state
and every cell uses G. No preparation exception is silently swallowed.
Backend preparation is separate, with a total
295s guard. Actual nested mutable-storage isolation, CUDA capture behavior and
combined preparation duration remain H200 gates; stub tests do not prove them.

The CLI rejects an oversized emitted production file. Its `--matrix` mode
writes a reproducible source-size audit for all16 masks, including failures.
It never dispatches a check or submission. A source-size pass alone would not
authorize an unmeasured override.

## Exact reductions and remaining byte gap

The G dependency slice preserves versioned eager aliases, final global
references, mutable-container history, startup tests and all license/NOTICE
comments. The only folded active Python branch is proved unreachable because
`GLE_T` is an unmodified empty dict; independent active `GLE_FRT` remains.
Unfactored result: 854399 bytes. Earlier smaller estimates that omitted
interstatement notices are superseded and must not be used.

Readable exact-string factoring stores common CUDA spans once and concatenates
them. It saves133874 bytes and produces720525 bytes, SHA256
`a6f97289c7abc8e49e7b7943e63f9b8138a8682988bb4c1b2603e3704fdc6464`.
The four reconstructed native source strings are byte-identical and the
remaining AST is unchanged. Both G main and helper0 are active; deleting
helper0 would break supported short paths/preparation. See the independent
[literal audit](2026-11-01-public-hybrid-literal-factoring.md) and
[G closure proof](2026-11-01-g-production-size.md).

The owned generator retains exact151256-byte ONE and121690-byte main51 MoE
literals. It removes unrelated native registrations/functions and inactive
projection families; the128 WD and256 boundary closures retain their selected
bodies. Actual raw backend sizes are389440 bytes (verify16),345746 bytes
(verify32),399155 bytes (both).

The complete readable transport and dispatcher produce the following actual
payloads. Every nonzero mask fails `<1000000` bytes:

| Fixed override | Mask | Complete bytes | Minimum additional bytes to remove |
|---|---:|---:|---:|
| Default exact607e | 0 | 869076 | 0 |
| Verify16 sliding only | 1 | 1115832 | 115833 |
| Verify16 full only | 2 | 1115832 | 115833 |
| Both verify16 cells | 3 | 1115832 | 115833 |
| Verify32 sliding only | 4 | 1072104 | 72105 |
| Verify32 full only | 8 | 1072104 | 72105 |
| Both verify32 cells | 12 | 1072105 | 72106 |
| All four cells | 15 | 1125548 | 125549 |

All other mixed masks are1125547 or1125548 bytes. The one-byte difference
comes from the decimal mask literal. Every source SHA256 and the complete
sixteen-mask budget are in
`muse_decode_layer/profiles/verify-hybrid-size.2026-11-01/manifest.json`.
This matrix was generated from tracked sources; it requires no ignored input
artifact and performs no GPU operation.

These are measured UTF-8 payload sizes, not latency measurements or theoretical
lower bounds. No source-text compression or silent deletion of active kernels,
fallbacks, startup tests or attributions is an acceptable fit workaround.

## Validation and next decisions

All 19 CPU tests pass together in 8.140s: nine dispatcher, five G closure,
two factoring and three owned-closure tests. They cover all16 masks/all12
scored cells, metadata negatives, default byte identity, unsupported-weight
fallback/deadline coverage, separate module namespaces and per-call state
ownership through explicit host facades. The raw transcript is
`evidence/2026-11-01-verify-hybrid/host-tests.txt`. These establish source/host
contracts only.

The owned source review preserves the128 WD/SGlang boundary,256c1x512,
three pinned Lt closures and main51 routing/execution. Local CUDA12.8 device-only
compiles compare the selected pruned/original WD, boundary and router kernels:
776, 424 and1,664 instructions respectively, with identical ordered opcodes and
operands after normalizing offsets/local labels/generated self-symbol names.
Whole ONE and MoE source literals remain byte-identical. These adapted
NVRTC/PTXAS screens do not test PyTorch host extension linking or the exact
H200 judge compiler/runtime. No new occupancy, utilization or latency is claimed.
The independently scoped review is
[here](2026-11-01-owned-production-independent-review.md).
Final source/host test results and exact evidence pins are recorded with this
report; runtime correctness, actual resource admission and preparation remain
unqualified. The source budget fails independently of those runtime gates.

The full 16-row matrix reproduced byte-for-byte from the merged tracked tools.
Its LF-normalized JSON SHA256 is
`1b871c3644647a94e116542b9e66a0eb89925d855d38bef6717af353569faa1c`.
The first reproduction check caught Windows CRLF versus LF in the receipt;
explicit LF output corrected that serialization difference. Source hashes,
payload sizes and all pass/fail results were unchanged. Reproduction and the
unchanged private diagnostic pin are recorded in
`evidence/2026-11-01-verify-hybrid/reproduction.json`.

Priorities and acceptance criteria:

1. Preserve the existing reset sequence: repaired six-load G stage comparison,
   then the independent fixed-natural-routing short-path trace. The four-cell
   verify diagnostic remains behind those checks, subject to runtime budget.
2. Before interpreting a four-cell win as actionable, identify a concrete
   production representation below1000000 bytes with exact ownership, packing,
   rounding and ordering proof. The current closure/factoring representation
   fails that gate. A different source reduction requires a separate reviewed
   hypothesis; this result does not prove that every representation is too big.
3. If that representation exists and the private cell benefit is repeatable,
   check the actual single-arm production file on H200: official13 correctness,
   fresh replay/routing/cache checks, resource admission and preparation<300s
   (local guard295s), then representative full-six-load comparisons.
4. Promote only under the registered full-score/noise/regression gate. A
   two-load event ratio, conditional score projection or isolated kernel win
   cannot replace the actual six-load scoring function.

Stop/pivot: stop this source-pruning pass at the demonstrated overlimit matrix.
Do not retry losing bytes, weaken numerical/ordering rules, substitute a partly
mixed pipeline and transfer the frozen diagnostic's result to it, or spend an
expiring public slot on a speculative hybrid. Retain607e and the private pins.
No new measured kernel bottleneck or resource-utilization result is claimed.
