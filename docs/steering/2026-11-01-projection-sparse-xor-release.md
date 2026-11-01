# Single-arm owned sparse-XOR measurement artifact

2026-11-01. Root requested this separate production artifact after the
paired diagnostic was prepared. Base `79d5d2b`. No GPU/public dispatch by
this worker; root retains the user's standing measurement authorization.
**Source/CPU candidate prepared, not an established performance winner.**

Frozen file `.build/today-proj/sk-xor-release.py`:321640 bytes, SHA256
`685ffab5d1614343d3c46d505967ae6817e3784736297018c78984354333f765`.
It is below1MB. The manifest is retained under
`evidence/2026-11-01-projection-sparse-xor-release/manifest.json`.
Reusable tools are `proj_sk_xor_release.py`, `proj_sk_xor_release_body.py`
and `proj_test_sk_xor_release.py`. The builder reconstructs exact raw C2
through the tracked restorer and verifies both C2 raw5610b8a6/LF7f4fd58c
and reviewed prototype d23aa17b. New CUDA remains exactly the prototype's
`8cdbb4a99f226beb3f96ca662ae9d260bb7e002ef697337d8f80c075ccaa5068`.
No new kernel or memory-ordering change is introduced in this release wrapper.

The math, layout, input, capture and immutable-cache-key helpers are copied
by AST at generation time from the audited owned paired harness, BODY SHA
`92b16040b139ead412cc40c1d81e1f2ea36137ea7f22870036e5d6f06c1c1ba0`.
No payload runtime file/git/reference import occurs. CUDA/CPP are readable
multiline literals. No private timing benchmark, report codec, score model,
comparison rounds or candidate timing selection is included.

## Fixed timed behavior and qualification

The original C2 prepare runs exactly once. One additional native module
contains control/treatment instances; controls serve resource and math
qualification only. The existing C2 prepare still performs its ordinary
unchanged autotuning. There is no new treatment autotuning or performance
gate inside this artifact. Its actual timed `kernel_fn` is byte/AST unchanged:
it reads the prepared pick dictionary as before.

After every qualification gate passes, fixed picks replace these ordinary
buckets, not merely the isolated scored shapes:

- qkv_gate/unpooler: NF at M2..7, full at8; NF9..15/full16;
  NF17..31/full32 (qkv32's original instance is already non-FULL).
- O/pooler: NF9..15/full16 and NF17..31/full32.
- All M1 routes, O/pooler M2..8, router, M>=33 and original general-M
  fallbacks retain their exact C2 callables. Thus only the intended10 of30
  official cells change;100 nearby keys share the same ordinary buckets.

Twelve immutable pack classes are reused by the audited explicit geometry
key. Each qualified full/NF control and treatment runner has private
diagnostics. Both full and NF resource queries must match independently
derived exact static/dynamic shared bytes and launch/max-thread attributes,
have zero local memory and positive occupancy; treatment resident blocks
must be at least its matching full or NF control. The expected slice8
metadata/shared padding is allowed; it is not silently assumed free.

Each target runs normal/probe, pure one-hot and outlier banks at M and M-1,
control and treatment, against FP64 GEMM rounded to BF16. This is12 checks.
Treatment capture adds one initial check and3 changed-input/output-poisoned
replay checks:16 per target,160 total. Original C2's stricter0.005 global
self-test tolerance and0.0005 probe gate are retained in addition to the
audited helper's row checks; treatment one-hot/probe outputs must be exactly
equal to the BF16 reference. Nonfinite references/outputs/metrics fail.
All40 new runner diagnostic buffers must remain zero.

Only after all tests succeed are the fixed picks published. Unsupported
finite/sparse packing is a legal per-bucket original-C2 fallback, reported
explicitly. Resource, math, incomplete coverage, timeout or diagnostic
errors throw; they do not select a measured winner or silently waive a
gate. Qualification/override counts distinguish a full10-target treatment
from partial/all fallback. No input/output/captured graph is retained in
returned state: only selected runner closures/immutable weights and their
diagnostics plus scalar metadata survive. Plain prepare-summary lines
report routes/resources/checks; there is no benchmark payload transport.

The295-second guard covers the normal C2 prepare, module build, packing,
inner tests, final diagnostics, publication and summary. A blocking compile
cannot be interrupted by Python, so actual completion within the external
300-second limit remains a material unresolved risk.

## Executed CPU gates and remaining risk

Five real-TorchCPU host-facade tests pass in0.609 seconds. They execute the
actual new prepare orchestration and actual original C2 timed entry with
explicit native/resource/graph stand-ins. Tests confirm1 normal prepare,
1 extra module,160 math checks,10 captures/30 changed replays,110 installed
keys, and actual dispatch for every M1..33 plus128/256/257 on all five
matrices. They verify unchanged callable identity for every untargeted
M1..256 key and all20 unaffected official cells. All/partial unsupported
packing falls back explicitly. Stale replay, bad admission, diagnostics,
nonfinite reference and outer/inner deadline negatives fail before pick
publication. Source tests ensure one original kernel_fn and no added
comparative timing/codec helpers. These tests do not compile the host
extension, execute SK/TMA/HMMA, or measure GPU performance.

Prior prototype evidence remains applicable to identical CUDA: exhaustive
lossless CPU packing/bit transport, independent source/lifetime review,
matched local sm90a controls and spill-free candidate code. A separate
RTX5070 probe exercises only the repair helper. Full H200 torch-extension
host compilation, every actual resource/admission/math/graph gate and
prepare duration remain unproved until an authorized official run.

The extra work is bounded to one module and12 immutable pack classes,
not four full C2 preparations. That reduces avoidable setup work but is
not a measured prepare-time forecast. Root must review exact bytes and
decide whether to use a single-arm scored measurement. A passing official
score is then actual candidate evidence, not a paired causal comparison;
the separate full30 heldout/null comparison remains the stronger promotion
gate. No top10/top50 gain or layer-port benefit follows from this artifact.
