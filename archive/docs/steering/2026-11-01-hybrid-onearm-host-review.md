# Independent single-arm hybrid startup host review

**Scoped source/host clearance** for exact candidate `4ad326c3451d1b9a7ba22e6c89a2b06ebd8f565c20300abd9fc0d3b0a35001bc`,980787 bytes. Seven independent CPU tests pass. No agent GPU, official call, public submission or queue action. This is not the old pure-comment packing clearance and supplies no speed claim. Parent owns whether to use an authorized single-arm public measurement.

## Exact source and public scope

The actual emitted file imports separate G/owned module namespaces, preserves the original production `kernel_fn`, and adds preparation-only tests. Actual whole-file execution with explicit external-import/native-build stand-ins resolves these Python globals. Its G backend remains `0fadbabe…`; owned MAIN51 device literal is exact repaired `fa97606f…` (checked cooperative host launch + fatal timeout), and ONE device literal is unchanged. The attached [review manifest](evidence/2026-11-01-hybrid-onearm-host/review.json) pins every full value. Resource getters add only native host APIs/declarations/exports and new cache labels, not device math.

The new suffix contains no event benchmark, elapsed-time selector, comparative arm, private report codec, autotuner or runtime source-file reads. `time.monotonic` only enforces295s. The fixed production mask15 still selects the four verify16/32 sliding/full routes by the exact metadata predicate; other scored/general inputs use G. An unsupported immutable owned scale bank retains G-only preparation and prints an explicit selected0/fallback record, rather than claiming four-cell qualification. That fallback does not qualify the intended all-four native hypothesis; the owning measurement must print selected4 and full counts. Existing inherited G preparation/startup is retained, not rerun by this CPU mock.

## Actual orchestration exercised

`tools/hybrid_onearm_host_review_test.py` executes **the complete emitted artifact**, including its real loader, normal dispatcher, qualification helpers and `_qual_oracle` clone. Only native/model arithmetic, backend preparation, missing SGLang/Triton/TVM imports, CUDA streams/graphs/synchronization and resource services are replaced. Real TorchCPU BF16/FP32/FP8 tensors carry full8192 hidden shapes and the unmodified public metadata predicate. Short fabricated CPU caches/inputs are explicit host-test stand-ins, not a natural-routing distribution or H200 math oracle. The real routing, row/global error and cache checkers plus finite/adaptive guards execute; their reference arithmetic is substituted.

The successful flow observes:

- One original production prepare (G once and owned once), then exactly one independent private owned prepare. Qualification uses the normal public dispatcher and never calls the G kernel on its four intended cells. It returns the original production state, not the private test state.
- Four cells ×two fixed seeds:48 full validations,8 graphs,24 observed fresh/base/fresh poisoned replays.
- The real component call targets `_moe1_run` ordinal10,PDLFalse, BF16[M,4096], global IDs32..39 with local_start0 and exact-zero output checks. It observes40 empty checks/8 empty graphs and32 occupied evaluations immediately following empty work, without resetting expert counters.
- Production/private mutable tensors and NN state are disjoint; Lt default-captured workspaces receive a separate alias check. No shallow function-closure clone is used. Private WD diagnostic, expert `ctr[8]`, cache/output poisoning, cache slot uniqueness, graph input retention, private warm stream/join, final cleanup, TF32 restoration and dequant-reference cache clearing all execute.

Negative cases reject stale full replay; stale empty replay despite prior zero output; nonzero empty output; failure of occupied work immediately after empty; output/reference NaN; routing/cache corruption; whole-state and Lt workspace aliasing; WD/expert diagnostic bits; malformed,nonresident,nonfinite or local-memory resource vectors; real early/late deadline predicates; and a propagated CUDA synchronization error. Unsupported scales exercise the G-only branch. No threshold or checker was weakened to make the fixture pass.

## Actual resource getter API review

The generator's getter definitions were read directly and compiled as C++17 with controlled CUDA-service mocks in `hybrid_onearm_resource_host_test.py` ([exact mock](evidence/2026-11-01-hybrid-onearm-host/resources.cpp)).

- MoE selects the same `moe1::moe_main_kernel<51>`, `sizeof(SmemMain)` and `kNTM` used by the live ordinal10 launcher.
- ONE selects `one::attn_one_kernel<8,3>` and `<8,51>` with the actual `Instance<8,MODE>::NT/BYTES`. The new getter explicitly checks its own dynamic shared-memory attribute call before checked attributes/device/cooperative/occupancy queries. It does not rely solely on old `Instance::prepare`'s unchecked initialization calls.
- API order/registration/declarations match:10-int MoE row and two10-int ONE rows, with fields regs/local/static/dynamic/threads/maxthreads/blocks-per-SM/SMs/cooperative/resident-capacity. The body requires zero local bytes, positive valid launch dimensions, cooperative support, actual residency capacity, and positive cached ONE residency no greater than its getter result.
- The exact getter mock returns three correctly ordered vectors and rejects all12 injected API error branches exercised independently here. Owner additionally tests its own getter branches. No CUDA error is treated as an optional numerical fallback.

These mocks validate host source/control and argument wiring, not the actual CUDA/Torch host link, H200 attributes or graph admission. Resource values in the fixture are controlled inputs, never reported measurements.

## Reproduction and remaining gates

From `tools`, use the TorchCPU environment with explicit paths:

```text
OKC_REPO_ROOT=<root> OKC_QUALIFY_OWNER=<owner/tools/prod_size_qualify_body.py> OKC_QUALIFY_ARTIFACT=<frozen-file> OKC_QUALIFY_BUILDER=<owner/tools/prod_size_qualify.py> python -m unittest hybrid_onearm_host_review_test hybrid_onearm_resource_host_test -v
```

Observed result:7 tests PASS in8.528s (CPU only). No further broad expansion is needed for this scope. Auditor's separate source/math/cache/provenance review is `13e70c4`; previous owned progress source review is `e57dd214`.

Remaining H200 gates are decisive: native host compile/link and installed API compatibility; actual cooperative MAIN51/ONE resources and capture; full model reference, routing/cache, fresh/empty/restore checks and exact counter coverage; complete preparation under295s; and all13 official correctness/all12 normal benchmark/profile rows with no flags. Startup has no comparative timing and cannot establish speed. The combined preparation budget cannot be forecast by summing archived source-specific setup times, and a running compile/reference operation cannot be interrupted by its next Python guard.

The parent reader must reject incomplete coverage, fallback-only qualification of the intended native candidate, any CUDA error or missing resource vector. A correctness pass with a lower full-layer score remains a loss; isolated target movement is insufficient. Do not change the two frozen G reset diagnostics or infer an automatic retry/promotion from this source review. Attribution/notices and fixed timed-path policy remain retained; source fit alone is not performance readiness.
