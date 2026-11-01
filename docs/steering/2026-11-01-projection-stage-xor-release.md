# Stage-local single-arm qualification artifact

2026-11-01. Built after prototype/evidence commit `e27c86d`; this is a
separate production qualifier, not the failed global-record release.
No GPU check or public dispatch was performed by this worker. Root owns any
measurement decision under the user's existing authorization.

Exact `.build/today-proj/stage-xor-release.py`: **450781 bytes**, SHA256
`de1111b98072f5251ff9b2edd5d453cc22eb8fe1710a943c60922909cc796229`.
Reproduce with `tools/proj_stage_xor_release.py ROOT OUTPUT` (optional
`--helper-body PATH` for the audited paired helper source). The generator
reconstructs tracked C2 through `proj_sk_xor_control.restore`, regenerates and
requires prototype SHA `b726aebd42c0749d3ff736fdb72d667d1143dce90819fc45b785580a1832bc2e`,
and copies only audited helper AST nodes at generation time. The payload
does not read source files at runtime. Its CUDA remains exactly the
stage-local prototype's `998b5efbf26f762cefdcc308b23629fa51dbbdc36569d0a466077aa44a4344da`.

## Fixed timed path and prepare gates

Run the normal C2 prepare once, then build one extra module using `sxs_build`.
That module includes C2, failed global-SX and stage-local instances; the
qualifier selects only C2 variant 0 and stage-local variant 2. It never runs
or times the global variant. No comparative timing, score codec, event-based
benchmark or new autotuning enters prepare. The existing C2 tuner is retained.
The original timed `kernel_fn` still reads `state['pick']`.

The target and neighbor policy is identical to the prior release protocol:
ten official SK8/16/32 targets plus 100 ordinary nearby keys. QKV/unpool
M2..7/9..15/17..31 use the matching NF instance and M8/16/32 use FULL where
applicable. O/pool start at M9. M1, O/pool M8, router, all M>=33 and all twenty
unaffected official cells keep the incumbent callable.

Capacity, row-map-only table ABI and stream bytes come from the exact
stage-local packer. The pack cache uses weight identity, representation,
row count/balancing and complete immutable pack geometry (twelve classes).
Before creating any runner, derive and reject dynamic extents outside
0..227 KiB. Then query actual FULL/NF CUDA attributes and require zero local
memory, exact static/dynamic/shared-thread derivation, matching max-thread
attribute, and candidate residency at least the **C2** corresponding FULL/NF
control, with at least one block. Capacity is printed with each route.
Resource or progress failure is fatal; it is not a fallback or an opportunity
to remove a target after observing results.

For every offered target, retain the prior 160-check protocol unchanged:
normal, onehot and outlier FP64→BF16 oracles at M and M-1 for both C2 and
stage candidate (12 checks per target), then one captured candidate result
and three changed-input, output-poisoned replay checks. Total 160 math checks,
10 captures and 30 fresh checks. The generator verifies the actual math-gate
AST equals the previous release's under identifier/diagnostic-label renaming:
finite outputs/reference/metrics, C2 .005 tolerance and .0005 probes, and
bit-exact candidate onehot probes are retained. Private diagnostics remain
disjoint and must all be zero.

Unsupported weight packing explicitly retains C2 and reports fallback
coverage. Successful override installation occurs only after all offered
targets and final diagnostics pass. The report records actual overridden
targets, fallback count, installed keys, math/capture/fresh counts, capacity,
arena bytes and four resource vectors per route. No input, output, capture or
oracle buffer is saved into production state. Only immutable packs, selected
runners, diagnostics and scalar qualification metadata survive. The entire
prepare, including original C2 prepare, is guarded at 295 seconds.

## Offline validation and remaining gap

`tools/proj_test_stage_xor_release.py`: five real Torch CPU facade tests pass.
They exercise actual prepare/qualification code and the original C2 entry
with native CUDA replaced by explicit CPU matrix-product stand-ins. Coverage
includes all M1..33 plus arbitrary-M fallback and all thirty official cases;
ten overrides/110 keys/160 math/10 captures/30 fresh; complete/partial packing
fallback; stale replay, admission, diagnostic, nonfinite-reference and
deadline negatives; and invalid analytical dynamic memory failing **before
any runner**. Source tests verify all sixty configured IDs select exact
geometry/precision and the intended STAGE/SX/FULL bits. They do not emulate
TMA or validate actual CUDA runtime execution.

The prototype's source/lifetime proof, independent actual packer tests and
all 45 locally compiled sm90a entries remain applicable without a device-body
change; see [the stage-local report](2026-11-01-projection-stage-xor.md).
Actual H200 host extension compilation, attributes/admission, all numerical
gates, setup time and performance remain unproved. The first global-record
release's 123.33-second prepare is historical evidence, not a bound for this
larger module or its new packing work.

Stop if resource/math/progress/prepare gates fail. If root measures it, use
the actual full-30 scoring function and explicit coverage; recovery from
global SX's 1.39412 loss alone does not beat C2's 1.527806 baseline. No claim
of a repeatable gain or top-10/top-50 feasibility follows from these offline
gates. No public layer port is included.

## Independent host clearance and terminal-analysis preparation

MoE commit `d276cc8` independently cleared exact `de1111` source and reran
all five real Torch CPU facade tests. Its additional RTX5070 helper-only
bit test covers the exact shared repair helper; it does not establish H200
kernel math, admission or performance.

Root registration `a08f501260ea7e9625c01aa4eeecda0934afa7f9` dispatched the
same exact file as public projection job `dfcfdd01d6f04525`. At this addendum
the worker has not received a completed receipt and reports no run result.
Root owns waiting, receipt verification and archiving. No additional official
call is made by this worker.

`tools/proj_stage_xor_release_analyze.py` is separate from the original
global-record analyzer and takes explicit receipt/log/source/job paths. It
requires the owning completed `muse_bf16_proj` run, exact release source pin,
clean official tests/checks, thirty unique cells, six official load sums and
exact score reconstruction. It parses the actual `SXSRELEASE_*` capacity,
arena, fallback and resource fields; compares each stage FULL/NF admission
to its matching C2 control; derives stage bytes from source layout functions;
and verifies reported qualification/neighbor counts. It supports explicit
partial packing fallback rather than silently treating fallback as stage
coverage. Outputs are UTF8 JSON/CSV with thirty rows, six loads, all resource
vectors, raw/adjusted/offset data and input seeds.

The primary comparator is original C2 `9eba96894ee1402e`. Any ten-cell
replacement into archived C2 remains clearly labeled as a cross-job
sensitivity scenario, not a paired test or official hybrid score. The failed
global version is secondary context only. The analyzer contains no automatic
winner claim or preset closure outcome.

Four synthetic tests pass: exact six-load reconstruction; explicit fallback
totals; foreign/running receipt, duplicate cells, nonfinite timing and score
mismatch negatives; capacity/arena mismatch, C2 admission loss and missing
coverage negatives. These tests produce synthetic values only, not an H200
measurement. Run analysis only after root supplies the terminal receipt:

```text
python tools/proj_stage_xor_release_analyze.py --job dfcfdd01d6f04525 --receipt RECEIPT --control C2_RECEIPT --log BUILD_LOG --source .build/today-proj/stage-xor-release.py --output OUTPUT_DIRECTORY
```
