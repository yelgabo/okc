# Four-cell layer comparison, compiler audit and private attention qualification

November 1. Continues the [completed 06:30 H200 round](2026-11-01-0630-measurement-outcome.md),
published as af3b550. No old locality, observer, C3 or projection-stage trial has
been reopened. Current production files remain unchanged. The top10–50 goal
remains active and unachieved.
Source/audit checkpoint before these root additions:59accfb; attention evidence
is the owningc79ddddb840a4fb7 receipt and its exact frozen86684555 source.

## A larger, policy-correct layer comparison now fits

The private G-versus-owned package now covers verify16 and verify32, each in
sliding and full context. It is **968389 bytes**, SHA256
`2bad52ceca9ef72d73baa539dc9db0c776a51b6009a4e4a4f8a8d9b725761c0f`.
[Generator](../../../tools/dl_verify_pair.py), [slices](../../../tools/dl_verify_pair_slices.py),
[manifest](../../../muse_decode_layer/profiles/verify-four-cell.2026-11-01/manifest.json).
The earlier 850769-byte single-cell package remains byte-identical and available.
This is a source/host qualification result, not an H200 comparison or a new score.

Independent review caught two meaningful policy errors in the first extension:
the owned128 path uses the selected QKV wide kernel, not Torch matmul, and uses
the SGLang attention-to-FFN boundary, not the owned c1x512 boundary. Both are
restored from the owning cb71 source, selection summary and profile rows.
[Review and corrected-package addendum](2026-11-01-verify-four-cell-slice-review.md).

The corrected owned128 path selects b1n128c3s8h QKV, SGLang boundary, FP32
concatenated pooler/router, native MAIN51, pinned layout3 unpooler and the original
FFN boundary. Owned256 retains its QKV/unpool Lt pins and c1x512 boundary. The
G arm retains all eight high-load Lt entries, native SC6/PRMT and ATT7, with
32-warps at128 and16 at256 for its row boundaries. Native device literals are
unchanged; the additional owned projection literals are exact copies of their
existing source. No fences, arithmetic or memory-ordering pattern were removed.

Seven CPU tests pass, including actual four-cell G host arguments, all three Lt
plan shapes/closure captures, policy selection, native literal equality, global
binding completeness, module/Lt workspace alias negatives and lossless report
transport. The default one-cell artifact's byte pin is also unchanged. Both
owned states are separately prepared, including WD diagnostic state and Lt
workspaces; G uses the reviewed guarded-storage clone. Captures retain their
input allocations. Lazy SGLang module tensor ownership is now checked too.

The runtime requires all four cells on two prespecified original/fresh banks,
96 oracle validations, exact natural expert IDs on both banks, WD/MoE spin flags
clear and all guard words intact. Each attention workspace warms before capture.
Preparation must finish within295 seconds. **Actual H200 linking, residency,
memory use and preparation time remain unmeasured.** These are release blockers,
as is any natural-routing mismatch. The current file is never a public submission.

### Conditional six-load value for each cell

From the owning607e adjusted medians, let C=46*t_sliding+16*t_full for its load.
If an event-time ratio R=owned/G transfers into a real hybrid on one cell with
weight w, while the other11 cells stay fixed, the conditional full-score factor is
`(C / (C + w*t_G*(R-1)))**(1/6)`. The report emits all four duplicate cross-ratios
for each cell; these are explicitly conditional transfer scenarios. The complete
six-load score is not measured by this four-cell experiment.

| Cell | Score factor for a hypothetical1µs cell reduction |
|---|---:|
|Verify16 sliding|1.000459834|
|Verify16 full|1.000159775|
|Verify32 sliding|1.000344576|
|Verify32 full|1.000119758|

These are arithmetic sensitivities, not predicted savings. A winning cell can
rank a proposed hybrid, but does not clear it: the complete hybrid still needs
source/interface/precision review, transparent byte fit, preparation budget and
fresh whole-score qualification. In particular the diagnostic package excludes
the other four loads; it is not an already shippable six-load hybrid. Preserve
the two registered reset checks; consider this larger comparison afterwards,
subject to the runtime gates.

## Projection: actual active schedules were inspected before proposing a port

The [projection lifetime audit](2026-11-01-projection-wgmma-lifetime-audit.md) finds
three active owned WGMMA cells among30. Their loops already use wait1 and retire
the previous operand slot; wait0 is terminal. The twenty short cells use HMMA or
CUDA-core paths and the remaining seven large cells use Lt. There is no source
evidence for copying G's MoE drain fix into these kernels.

An isolated, fully pinned CUDA13.0.88 compiler/cache was installed. NVRTC13 and
PTXAS13 compiled the same22-entry sm90a input, decoded with NVIDIA nvdisasm13.0.85.
All three WD wait/resource tuples equal the CUDA12.8 results: four/eight/eight
HGMMA instructions, one wait1, one terminal wait0,90/58/90 registers and zero
spills. [Compiler manifests, logs and comparison](evidence/2026-11-01-projection-waits-cuda13/comparison.json).
This narrows the compiler-version gap; it is still a local NVRTC build, not the
official H200 NVCC CUBIN or opaque Lt kernel instructions. Reproduce with
`LD_LIBRARY_PATH=/root/.cache/okc-cuda130-88/nvidia/cu13/lib` and the compiler
tool's `--cuda13 /root/.cache/okc-cuda130-88` option. The exact package versions,
binary hashes and NVIDIA redistribution provenance are recorded in
[toolchain-install.json](evidence/2026-11-01-projection-waits-cuda13/toolchain-install.json).

The separate [mailbox audit](2026-11-01-projection-mailbox-scope.md) corrects the
assumption that a new async mailbox would replace global short reductions.
Selected CL1 SK and RD reductions are CTA-local; RT already uses async mailboxes
and a split exit barrier. The old failed push trial also used a dedicated receive
buffer, st.async completion and receiver acquire. No distinct new port or GPU
check is justified by that analogy. Neither audit authorizes a barrier removal,
stage/tile sweep or C3 repeat.

## Attention: private H200 qualification finished below the whole-score gate

Fresh [bounded public-source discovery](2026-11-01-attention-public-source-delta.md)
found no stronger new exact artifact; known strong public files and checked
research branches were unchanged. It did not repeat the previous109-handle scan
or use a team's newer score as evidence for its older source file.

The remaining VP2 signal is structurally distinct from the failed64-key generic
DSM complete-head variant: a dedicated owner mailbox with st.async completion
permits owner acquire and local-only reads without the late raw-KV alias barrier.
It does not save logical payload relative to that earlier cluster scheme, and
N128/QK/P changes remain bundled. There is **no qualified V1 wait-duration bound**.
The old approximately0.73µs positive point was unrounded and close to the
approximately0.722µs whole-score1% break-even. This is threshold uncertainty to
resolve, not a measured storage-only saving or projected speedup.
[Exact source/lifetime contract](2026-11-01-vp2-private-source-review.md),
[qualification and decision registration](2026-11-01-attention-vp2-bf16-requalification.md).

Root dispatched **c79ddddb840a4fb7** with exact412952-byte source SHA256
`866845554b3d694c40165fabbdfde6739f6fa22b1efb5abb6b5831f32e368f24`.
The job completed with16/16 official tests,12 benchmarks and12 profiles,
clean policy checks,640 private checks and all36 held-out timing rows.
The actual private comparator scored **2.594319659×**,95% CI
[2.555404864,2.633234453]. Its eligible official target profile names
`vfgns::b1vp2::vp2_kernel`, so the measured target did not silently use fallback.
[Full owning receipt](../../../muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.receipt.json),
[frozen-reader analysis](../../../muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.analysis.json).
Both source and independent host reviews cleared the frozen bytes. Eight host
tests pass, plus separate source/math/reader tests. Metadata predicates protect
the normal raw-pointer ABI; unsupported cases delegate to the existing owned or
SGLang paths. All intended private targets must actually hit VP2, with explicit
20 VP2/284 owned host-call coverage. All private state/input/cache owners are
separate, and BF16 output boundaries precede RMS on both sides. The FP32 judge
and BF16 baseline oracles remain distinct.

The normal **private** judged path uses VP2 on the eligible target. All36 private
held-out rows and640 numerical/cache checks completed. Actual warm/capture
coverage was20 VP2 and284 native calls. No
foreign standalone implementation is authorized for public adoption. Two active
CS16 clusters, checked launch attributes, zero local memory and the registered
SCH0 configuration passed runtime gates. VP2 uses97 registers,81984dynamic
shared bytes,256threads,2CTAs/SM admission and14active clusters. The owned
target uses82registers,90704shared bytes,256threads and2CTAs/SM admission.
These are resource/admission attributes, not measured achieved utilization.
Private preparation finished in102.041seconds.

The reader requires the raw terminal status, not normalized profile JSON. Old
fixed timing offsets and raw score factors are reported separately; neither
selects the decision alone. Before dispatch, the decision was registered to use
this job's actual per-cell raw/adjusted offsets for a common calibrated sensitivity,
then require ≥1%,2×null, each seed and no regression, with all raw cross-ratios
positive per seed. Nonpositive calibrated times fail closed. The official score
and this paired sensitivity are different observations. Even a positive signal
requires an owned implementation and its own full-score check before promotion.

The target's private raw-event means were10.784→10.168µs across three seeds and
duplicate arms, a5.71% cell reduction. The calibrated full-score cross factors
were1.006309–1.009060, below the registered1.01 threshold. Duplicate null was
0.002371; every per-seed gate failed. Raw signs were positive and the regression
gate passed. **The frozen decision is false.** It is not a qualified whole-score
gain or an authorized standalone code adoption. No owned rewrite, automatic
retry or copy/cluster/layout sweep follows.

Keep the comparator discrepancy scoped and visible: the old64-key/generic-DSM
owned experiment lost, while this complete BF16-output-matched bundle wins the
target cell. The N128/QK/P/storage bundle is not isolated and the whole-score
budget is below the gate. Reopening requires a distinct owned mechanism or a
measured portfolio with sufficient repeatable score leverage, not the old
unrounded point estimate. No new public action occurred. The best submitted
layer remains607e at2.126082676×. Attention has3 free checks remaining; the
layer/projection/MoE reset work remains scheduled for16:00 PST.
