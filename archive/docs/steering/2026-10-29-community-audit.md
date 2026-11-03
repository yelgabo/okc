# Community research audit and updated steering, 2026-10-29

## Decision

Finish integrating the measured component gains, close the specific correctness gaps, and run a repaired whole-layer comparison when check quota returns. Keep the current fused/hybrid design as the control. The uploaded research supplies useful audit questions, but its proposed experiment sequence is behind this repository's results and assumes quota that is exhausted.

The layer remains **1.964086x**, with 13/13 judge tests passed. Attention's newer **2.504129x** and projections' **1.443879x** are component scores. They do not establish a faster integrated layer. Mode 37, sk8w3s, QKV8, the newer cgemv, and attention DCARRY/PK16/VHOIST all await layer integration at the evidence commit.

Priorities are concrete:

1. Freeze the incumbent and prepare the five port packages below. Repair comparison acceptance and the missing FIN8 targeted replay coverage.
2. Inspect generated attention code for the reported timeout/trap compiler pattern. Audit PDL by data provenance and preserve timeout failure reporting and required fences.
3. After quota reset, validate the integrated layer and compare it with the incumbent and a pinned public implementation under identical conditions.
4. Give the already queued fat-MoE mode 43 mechanism one bounded comparison. Defer a new small-token sweep, asymmetric T32 design, and generic attention-prologue work until whole-layer evidence justifies them.

These are recommendations for the implementation session. This audit changed documentation only and launched no check or scored submission. Publication authorization does not authorize `okc submit`.

## Scope and evidence labels

Repository evidence is pinned to [`21be06df87e569cb514c3a951993a5d571b857d2`](https://github.com/yelgabo/okc/commit/21be06df87e569cb514c3a951993a5d571b857d2), fetched from `origin/main`. Live account evidence was first captured at **13:43 PDT, October 29**, with component-board reads later in this audit. The [capture record](evidence/2026-10-29-community-audit/capture.json), [quota](evidence/2026-10-29-community-audit/quota.json), [recent jobs](evidence/2026-10-29-community-audit/jobs.json), and saved status responses make the observations available to other sessions.

The audited upload is [OKC Community Audit Verification and Judge-Run Execution Plan](evidence/2026-10-29-community-audit/uploaded-research.txt), preserved verbatim. Its SHA256 is `af3fb11f71a6def3a739edad29286e4dc61274915064cb3dd7e7558f8cb9e948`. Its embedded search citation IDs are historical annotations, not independently usable source links. Direct links below identify what this audit actually consulted.

- **Verified source fact** means inspected code, an official response, or an archived receipt. A receipt verifies what was recorded, not an independent rerun.
- **Recorded local measurement** means committed benchmark output or a run log. It retains the named workload, control and instrumentation limits.
- **Externally reported claim** means a participant or upstream author reports a result. Finding the original post verifies the attribution, not reproducibility on our code.
- **Proposal** means work or an expected benefit without a completed result for that candidate.
- **Completed** describes a specific implementation or test. It does not imply integration or submission.

A separate read-only reviewer checked the local source, pending ports, harness defects and rejection register. No fresh CUDA compilation, H200 timing, sanitizer run or replay soak occurred in this audit.

## Current scores and available quota

The [receipt summary](evidence/2026-10-29-community-audit/receipt-summary.json) is derived from the saved `okc status <job> --json` responses. The saved component-board responses confirm the current selected entries.

| Problem | Current board job | Speedup and 95% interval | Judge tests | Interpretation |
|---|---|---|---|---|
| Whole layer | `9b921febc7ad4140` | 1.964086x, 1.934624-1.993547 | 13/13 | Incumbent, source `a923f11` |
| Attention | `e9085c86d89449c2` | 2.504129x, 2.466567-2.541690 | 16/16 | Scored benchmark file whose judged path is the PK16/VHOIST candidate |
| Projections | `81254e8aed794eab` | 1.443879x, 1.422221-1.465537 | 9/9 | Candidate-only scored cgemv source `a35709c` |
| MoE | `6307d0c91c8d4ada` | 2.159764x, 2.127367-2.192160 | 5/5 | Scored mode-37 source `96f2c43` |

Two other results must stay separate. `75bca9775f6f4b0b` scored 2.006267x, 5/5, while its appended comparison established the three-slot candidate's paired gains. That headline is not a score for a newly integrated sk8w3s layer. Diagnostic check `ca43908f02874886` scored 2.174297x, 5/5. It repaired wait attribution; it is neither a scored promotion nor a new production optimization. [MoE run log](../../../docs/runs-moe.md).

Attention's candidate-only run `569b233dfe08414f` passed 16/16 at 2.449x before the 2.504x repeat. The repeat used a faster machine on several unchanged rows. Credit the paired PK16/VHOIST measurements and their repeated full-verify improvements, not the entire headline-score difference. [Attention run log](../../../docs/runs-attention.md).

| Problem | Checks left / daily limit | Scored slots left / daily limit |
|---|---|---|
| Whole layer | 0 / 20 | 0 / 10 |
| MoE | 0 / 20 | 0 / 10 |
| Attention | 0 / 20 | 1 / 10 |
| Projections | 0 / 20 | 4 / 10 |
| Prefill | 20 / 20 | 10 / 10 |

These are timestamped account observations, not reservations. Re-query before each run because other sessions share the quotas. The [official rules](https://openkernelchallenge.dev/rules) state reset at 00:00 UTC, which is **17:00 PDT on October 29** for this allocation. Do not redirect decode experiments into prefill quota or spend the remaining component scored slots simply because checks are exhausted. Existing explicit user approval must cover any scored candidate; otherwise obtain it after preparing a reviewable candidate.

The [overall CLI snapshot](evidence/2026-10-29-community-audit/overall-board.json) reports yelgabo at 229 of 1,612. It supplies ranks and labels its measure as rank across problems, but exposes no ranking formula. The [decode-layer snapshot](evidence/2026-10-29-community-audit/layer-board.json) also places yelgabo at 229, with rank 50 at 2.1885x and rank 100 at 2.1278x. These are board-specific observations. Neither the old 2.115x rank-100 snapshot nor 2x is a verified qualification threshold. Reaching the observed layer rank-50 score from 1.964086x would require about 11.4% more speedup, or 10.3% uniform latency reduction. This arithmetic is a scale estimate, not a forecast or an overall-ranking rule.

## Audit of the uploaded findings

| Research finding | What is verified or reported | Updated disposition |
|---|---|---|
| Bounded wait plus `__trap()` serializes WGMMA | [QuantIntellect's original post](https://guild.1nterface.ai/t/18?credible=all) reports nvcc 13.0, sm_90a, C7512, a 168-register cap and serialized HGMMA. This is participant evidence. Our attention source contains bounded trap paths in `mbar_wait` and `coop_barrier`. | Keep a targeted generated-code audit. Source presence alone proves no compiler defect. Our recorded local SASS work uses CUDA 12.8.1, so it does not reproduce the reported toolchain. Preserve fail-fast behavior. |
| Humming resource compaction | [`get_compact_moe_config`](https://github.com/vllm-project/humming/blob/main/humming/tune/sm90_policies.py) requires WNA16, non-dense WGMMA, warp specialization, eligible block/warp shapes and block-K 64. It accepts a refit only if tile, warp, residency and pipeline checks hold. Its single-warpgroup eligibility is broader than only the grouped branch. | Use the resource checks as a heuristic. Our tiny path uses mma.sync. Its three-slot experiment has already completed; integrate that result before another sweep. The source changes several coupled resource choices, so it also does not justify a universal one-parameter-at-a-time rule. |
| zhebrak above 2x | The [pinned archived receipt](https://github.com/zhebrak/muse-spark-kernels/blob/ff4d55ab669977f15874943044d20312d98a44d1/winners/muse_decode_layer/result.json) records job `3c4a470e51c4421e`, H200, passed, 2.014592x, interval 1.984374-2.044811, finished October 28. | Receipt verified, no matched comparison against current yelgabo. It supports studying a concrete alternative, not copying its architecture or treating its score as a current target. |
| zhebrak asymmetric MoE | The [pinned MoE source](https://github.com/zhebrak/muse-spark-kernels/blob/ff4d55ab669977f15874943044d20312d98a44d1/winners/muse_moe_w4a16/submission.py) contains separate GEMM entry points and shape-specialized dispatch. Its own summary distinguishes 1-7, 8-63 and 64+ token designs. | T32 asymmetry remains a proposal here. Defer behind integration and measured T32 attribution. The local staged loss was at T128/256; it cannot reject every T32 asymmetric design. |
| Attention prologue gains | [FlashInfer PR 6976](https://github.com/flashinfer-ai/flashinfer/pull/6976) reports 0.5-1.8% versus its immediately preceding version for two BF16, head-dimension-128 sparse-attention programs on H100. | Upstream measurements, not a Muse speedup prediction. Our current consumer/address improvements already have local evidence. Port those before inventing another prologue experiment. |
| PDL ordering | [FlashInfer 7525](https://github.com/flashinfer-ai/flashinfer/issues/7525) reports a non-coherent load and missing memory clobber on SM103a, without an observed wrong result. [7526](https://github.com/flashinfer-ai/flashinfer/issues/7526) shows pre-wait metadata reads in SM100f cubins and a related GB300 counter failure. | Useful provenance and ordering questions. Our observed layer waits already carry memory clobbers. A load is hazardous if its writer can still be live, not merely because its spelling is `ld.global.nc`. No corresponding H200 defect was demonstrated in this audit. |
| Projection L2 prefetch loses | [Sudhendra's Guild report](https://guild.1nterface.ai/?sort=top) reports 0-18% regressions. The warm-T1 0.3 us detail in the upload was not independently recovered here. | The local bulk-hint and real-load experiments already lost, with job IDs below. Keep those mechanisms closed on their tested workloads regardless of the external anecdote. |
| Replay-only bugs | [jakozaur's original post](https://guild.1nterface.ai/t/42) reports barrier-storage reuse, missing async-proxy ordering and atomic-workspace failures under queued graph replay. | Verify attribution, retain as external evidence. Our repository already has 2,000-replay tests. Extend the missing candidate/mode coverage instead of presenting replay validation as entirely new work. |
| Other public repositories | [NathenFernandes](https://github.com/NathenFernandes/muse-bf16-proj) reports 1.5186x; [operatorstack](https://github.com/operatorstack/open-kernel-challenge-solutions) publishes solutions; [infraax](https://github.com/infraax/okc-kernel-challenge/blob/main/HYPOTHESES.md) has a hypothesis log. | These sources are available, but this audit establishes no new locally measured advantage from them. Nathen's headline alone does not prioritize a port over our completed gains. |
| vjeux / ruwwww identity | The upload explicitly leaves exact file/blob identity unverified. | Keep it unverified. Do not depend on file-size similarity. Pin and hash the exact public file used by the repaired comparison. |

The research's fixed pipeline, from a small-T sweep to asymmetry to prologue to integration, should be replaced. It postpones the outcome that matters while repeating completed work. Integration and a controlled layer comparison eliminate several speculative steps. The strongest bounded research alternative already queued is mode 43, which addresses a measured wait boundary within the fused design.

## Integration manifest

`git diff a923f11 21be06d -- muse_decode_layer/submission.py` is empty. The layer file's SHA256 is `3e9ba3df33f4b8e6fc5d26e17a33ad1c9c0e3ebd669b87e89c8681ce1eae5b16`. The judge receipt separately records payload content hash `8de7099397c558e9ac1a100ec6736ee113fea0d4bec9ddafcd477dbe05010a7f`. Preserve both identifiers; they are different hashes and must not be conflated.

| Port package | Completed evidence | What remains in the layer |
|---|---|---|
| Fat MoE mode 37 | Scored `6307d0c91c8d4ada`; ordering review; paired improvement over 28. | `tools/dl_moe_inline.py` still lists modes 0/16/26/28. Add 37 to the mode and name tables, regenerate, retain the earlier valid candidates, and validate the layer's PDL preamble. |
| Skinny MoE sk8w3s | `75bca9775f6f4b0b`, 36/36 paired seeds faster across T1/8/16, layer and standalone routing. Layer-routing savings about 0.9/2.4/3.1 us. | Regeneration includes the new mode-0 skinny candidate. Confirm actual selection and replay it with mixed-size calls on shared scratch. Do not force its T1/8/16 result onto T32. |
| Attention DCARRY + PK16 + VHOIST | DCARRY checks `f64e8680b3e441be` / `bf49eceb15a64f81`; PK16/VHOIST paired `025ccaf9c57a4ba2` and scored `569b233dfe08414f` / `e9085c86d89449c2`. | Re-inline current `attn_one.cu` through `tools/dl_one_inline.py`. All three switches transfer without new ownership/interface fields. Keep the layer's dependency waits and fallbacks. Verify the source and generated switches. |
| Eight-token QKV geometry | `826cab6b81aa4f81`, source `fee7249`; paired saving about 0.66-0.69 us. | Layer still has the old qkv8 tuple. Regenerate matching packing and geometry together through `tools/dl_proj_inline.py`. |
| New single-token cgemv | `a35709c`; paired `f6c80b4ba11048f3`, scored `81254e8aed794eab`. QKV improves 0.45-0.54 us, pooler 0.86-0.96, slice 0.22-0.35 in the paired passes; o is inconclusive. | Port row-major byte/code planes, packer, helper, 32/40-register dispatch and device kernels together. Tensor shapes stayed the same while byte layout changed. The inliner also includes the pending qkv8 change, so inspect both rather than crediting one. |

Read [MoE](../../../docs/moe-port-guide.md), [attention](../../../docs/attention-port-guide.md), [projection](../../../docs/proj-port-guide.md) and the [latest projection port steps](../../../docs/runs.md). Generate twice to check idempotence and inspect every unrelated diff. Keep attributable package controls, then test one coherent combined release. Do not require every coupled ingredient to win independently in the old design; PK16/VHOIST and the new cgemv already have combined evidence.

FIN8, QKOV/EREL/PCHK/UNR2, PFAST, main-stream routing fusion, the earlier GEMV port, and large-token boundary selection are already in the incumbent. They are controls, not new ports. CUTLASS's standalone pooler gain is not an automatic layer port because the layer's large-token front uses concatenated pooler/router work. [Layer run log](../../../docs/runs-decode-layer.md), [rejection register](../../../docs/ruled-out.md).

No combined score is predicted here. The previous approximate 1.98x transfer estimate predates sk8w3s, PK16/VHOIST and the latest cgemv. Summing newer component deltas would ignore selection, overlap and interactions. The next decision needs actual six-load layer timing.

## Correctness and measurement work that remains

### Close FIN8's actual coverage gap

Existing replay tests cover several attention paths, and the incumbent passed the judge. However, the intended q_len8 FIN8 stress chose mode 15. The current `_one_stress` generates contexts around 1,500-5,000 and prints the plan without asserting mode 11 and eight copies. Its documentation claims stronger coverage than the test establishes. [Layer run log](../../../docs/runs-decode-layer.md), [layer source](../../../muse_decode_layer/submission.py).

Select an input that actually chooses q_len8, mode 11, eight copies, assert the selection, then run queued replays, fresh-input comparisons, workspace/counter checks and the supported speculative-length fallback. Update the test's documentation with the asserted path. Keep the existing test coverage, including judged full verify-1 and self-test case 18, distinct from this missing stress.

The cited replay post recommends bitwise comparisons, but allowed floating-point atomic order changes require task-appropriate tolerances. Use the independent reference and task bounds, strict same-candidate comparisons where justified, and exact counters/cache invariants. Report every retained output's check and batch synchronization interval. Two thousand replays are a coverage target, not proof of a memory-ordering argument. Since this Mac has no NVIDIA GPU, queued replay validation is not a zero-cost local step unless a separate H200 executor is actually available.

### Repair the whole-layer comparison before spending a check

[`tools/dl_make_bench_pub.py`](../../../tools/dl_make_bench_pub.py) still compares outputs only for seed zero and prints `ABCHECK` differences without enforcing acceptance. Rotating then reversing the arm order repeats AB for a two-arm comparison. Arms also receive the same `data` object, including writable caches. These issues remain unresolved at the evidence commit.

The next owner should implement acceptance against the independent reference and task routing tolerances, explicit AB/BA or seeded permutations, an ours-versus-ours null arm, and independent mutable cache/scratch state with identical starting inputs. Validate every measured seed, save routing agreement and local expert work counts, and add independent held-out seeds rather than relying on literal scored-case seeds.

Compare frozen incumbent, integrated candidate and one pinned public layer. Cover decode B1/16/32 and verify B1/16/32, both sliding and full, with cold-L2 graph timing. Print a compact pass/shape/policy manifest that survives log truncation. If the prepare limit forces `ABSTOP`, enumerate omitted shapes and split the remaining work into another check. A partial result cannot establish the whole-layer score.

Use load time `46 * sliding + 16 * full`, then the geometric mean of six baseline/candidate load ratios, as specified in [TASK.md](../../../muse_decode_layer/TASK.md). Preserve paired samples and distinguish judge-derived times from raw medians. The attention log demonstrates a bimodal SGLang baseline in decode-32 full. A changing derived row is not automatically a regression in our unchanged kernel. Profile timelines locate dependencies; sums of kernel rows do not yield an idle-time budget.

### Bound the compiler and PDL audit

Inspect the actual generated attention variants used by the candidate and layer, including mode 51. Record compiler version, sm_90a flags, warnings, resource use, and HGMMA/DEPBAR scheduling. Compare the trap-bearing path with the existing pipelined baseline. A 12.8.1 local compile is useful screening, but a nvcc 13.0 participant claim needs matching-toolchain evidence before reproduction is claimed. A missing C7512 warning alone is insufficient; inspect the instruction schedule too.

Do not remove traps or replace a bounded timeout with an unbounded wait as a performance patch. If a change is needed, preserve deterministic failure reporting, test timeout behavior and retain the ordering proof. Existing C7512 mixed-shape dispatch and C7513 divergent lane-0 release losses are different mechanisms, already recorded in [ruled-out.md](../../../docs/ruled-out.md).

For PDL, enumerate each pre-wait load, its writer, and whether that writer may still be live. Layer waits observed in source already have `"memory"` clobbers. `LO_HELPER` reads `local_start` with `ld.global.nc`, but that is an input rather than route-produced state; the load form alone does not establish the FlashInfer hazard. Static weights and independent setup can precede waits. Predecessor-produced activations, metadata and counters require the actual dependency chain. A passing stress test does not authorize removing a proxy fence.

### Pinned SiLU source check completed in this audit

The prior mode-37 review consulted a newer SGLang revision. This audit read the baseline's actual pin, `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. Its [activation wrapper](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/activation/activation.py) routes filtered SiLU through the default fast-math module. Its [architecture flags](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/utils/arch.py) enable fast math for Hopper. Its [device implementation](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/elementwise/activation.cuh) converts gate/up to FP32, performs SiLU and multiplication, and casts back to the tensor dtype on this path.

This closes the source-version question about BF16 boundaries and fast-math policy. It does not prove bitwise equivalence between compiled division and mode 37's reciprocal. Keep numerical validation and the [mode-37 ordering argument](../review-mode37-ordering-2026-10-29.md). File hashes and matching source lines are saved in [pinned-sglang-source.json](evidence/2026-10-29-community-audit/pinned-sglang-source.json).

## Completed experiments and closed mechanisms

These entries reconcile the upload with [docs/ruled-out.md](../../../docs/ruled-out.md). Preserve the tested mechanism, configuration and workload in every future rejection.

| Item | Completed evidence | Queue consequence |
|---|---|---|
| Fat wait accounting | `3f67a45`, `ca43908f02874886`; first try_wait is now timed. Mode 41 adds 7.0/6.1 us at T128/256; mode 42 adds 1.1/2.3. Production SASS identity was recorded. | Remove the repair from the queue. Waits are about 20% of stage-loop time, not zero. Per-warp waits are not additive SM or layer savings. |
| Skinny ring/refill | Three slots win in `75bca9775f6f4b0b`; refill-order changes and dropping the L2 hint lose. | Integrate sk8w3s. No repeat of the same ring/refill study. |
| Skinny ALU/barrier work | `e937013292b24906`, scale-decode and partial-buffer handoff are null; early release in `2c28a7256b004aa1` helps layer routing modestly but loses standalone. | No blanket producer/resource sweep based only on Humming. A new candidate must cross a demonstrated resource threshold and beat the integrated control. |
| Staged two-launch MoE | `e221858b020c4dee`, complete staged pair 277.3/311.6 us versus mode16 249.4/287.3 at T128/256, also recorded in the architecture tracker. | Keep this decomposition closed. Asymmetric T32 tiling is different and remains deferred, not disproven. |
| Fat volatile reload and paired stores | `c7db20e053254b52`, mode35 reduces spills but slows execution; `473d87d03de94e96`, mode39 improves an epilogue bucket but loses whole MoE. | Fewer registers or store instructions alone do not justify promotion. |
| Projection prefetch | `985732949c07474f` bulk hints, `34568f495b024c2f` real loads, no useful whole-layer win. | Zero planned quota for the same side-stream design. Reopening needs a different footprint or measured idle bandwidth. |
| Blanket PDL and boundaries | `179cecdac87b4872` / `e8b74a3519a94bc6` show no clear step gain from general early launch; `5285e58d978f42c6` / `7631043129f74b61` settle the boundary geometry. | Do not repeat either as a new lead. Existing MoE pre-wait hints are a different bundled change, not an independently measured prefetch win. |
| Attention copying and pair balancing | KVFAST, block-index weighting and arrival-order weighting are recorded losses/nulls. Once-path diet loses in producer modes 51/115. PK16/VHOIST already reduces the consumer instruction cost. | Port the winning code. The latest cold public gap is only about 2.2/3.5 us at full verify B16/B32 in `e9085c86d89449c2`; an unbounded prologue or balancing campaign is poorly justified. |
| Wide concat front | `f88e278`, arithmetic using `8bea43d605c74546` puts the idealized proposal at 24.4 us versus 24.0 us measured cuBLAS control. | Declined analytically, not a measured challenger loss. Reopen only with a materially faster streaming kernel. |

## The bounded research slot

The [fat-wait mechanism report](2026-10-29-fat-moe-waits.md) already proposes modes 43/44. They are unimplemented at this snapshot. Mode 43 combines smaller code groups with next-stage waits and the first code group under the current stage's last WGMMA group. This is a latency-hiding hypothesis grounded in the repaired measurements, not merely mode35's losing volatile reloads or a claim that register pressure is the bottleneck.

Retain its gates: local SASS/resource screening first, no C7512/C7513 serialization, preserve production controls and ordering; then at most two MoE checks after reset. The first paired run targets T32/128/256 with layer routing and six seeds. Require at least 2 us mean improvement at T256, at least five of six seeds faster, at most 0.5 us mean regression at T32/128, and accepted numerical comparisons. Mode44 must reduce average cycles per wait by at least 30% against mode41's 209-231 cycles to support the proposed mechanism. If timing improves without the expected wait reduction, repeat before promotion and label attribution unresolved. If waits shrink without a timing win, stop that mechanism. The final candidate still needs held-out/skewed routing and layer evidence.

This is the real bounded test of the strongest current alternative to simply porting gains. Keep its budget separate from integration. A loss or tie closes this hoist depth; it does not close all pipelining. If the repaired whole-layer comparison identifies a different dominant cost, revise the priority rather than forcing the queued idea into the release.

A genuinely different T32 GEMM1/GEMM2 layout remains a reserve option. Before building it, show which phase loses on held-out T32 routings, how the candidate changes bytes/residency/readiness, and why its launch/finalization costs can fit within the recoverable layer time. One coherent asymmetric pair can be tested without demanding that each ingredient win in the old design. No such new experiment is claimed completed here.

## Execution and stopping plan

Before reset, the implementation owner can freeze source and generated-code identifiers, prepare the integration, repair the harness, add explicit test-selection assertions, and inspect compilation. These actions do not require a scored submission. A failed ordering argument or incorrect output blocks promotion even when a timing improves.

After reset, re-query account quota and coordinate a shared reservation. Proposed initial reservation, subject to actual capacity:

| Allocation | Purpose | Limit |
|---|---|---|
| Layer check 1 | Integrated candidate, required self-tests, selected-mode assertions, queued replay coverage and all 13 judge tests | Stop on any failure; retain incumbent |
| Layer checks 2-3 | Repaired incumbent/candidate/public/null comparison across all 12 scored combinations; split only for time limits | Missing shapes remain explicitly incomplete |
| Layer check 4 | Confirmation of the release file or an interleaved control if exact-source deduplication prevents a useful repeat | Whole-layer gain must exceed the measured null/noise and survive per-load review |
| Layer checks 5-6 | Reserve for integration repair or follow-up evidence | Leave unused unless needed |
| MoE checks 1-2 | Queued mode43 comparison and held-out gate/repeat if warranted | Independent of layer reservation; no automatic submission |
| Attention/projection checks | Only a specific port regression or correctness gap needing isolation | No automatic rerun of already established component results |
| Scored submissions | Promotion of a concrete validated source | Zero automatic allocation; explicit user approval and live quota required |

The reservation is a proposal, not consumed quota or an instruction to run every slot. Runtime limits may require repartitioning coverage. The upload's 5-7 layer checks and two scored runs are not an entitlement. Do not spend the last confirmation capacity on another sweep.

For check dispatch, preserve the repository's job-identity rule: submit with `okc check <file> --no-wait --json`, capture that returned job ID, then read `okc status <job> --wait --json` and verify the problem and source hash. Save each result, selected policies, failures and fallbacks in its run log. Record every rejected experiment in `docs/ruled-out.md` with its precise scope.

Promotion needs correct outputs, justified ordering, explicit changed-path replay coverage, a reproducible whole-layer benefit, and an explanation for any material per-load regression. A 2% row change is a useful investigation trigger, not proof of failure or permission to ignore smaller systematic losses. If the combined candidate is neutral, use attributable package controls to isolate the interaction. Stop after roughly three weak experiments and reconsider the mechanism and workload assumptions.

The next handoff should contain the exact release source and payload identity, generated-source hashes, actual selected policies, coverage matrix, per-load paired results, and the next measured bottleneck. Until those exist, the current layer score stays 1.964086x.

## Audit limits and handoff

The source audit and live receipts establish the current state and the work still required. They do not establish fresh GPU correctness, a new integrated score, compiler reproduction under nvcc 13.0, exact public-source identity, or a current overall qualification formula. The public web leaderboard was unavailable in the fetched page, while the CLI returned account and board data successfully.

Older steering and architecture documents are dated snapshots. This report supersedes their active queue, while their original evidence stays intact. In particular, the fat-wait repair is completed, modes 43/44 remain proposals, and the latest attention result adds PK16/VHOIST to the pending ports. Read this report through the [shared steering index](../../../docs/steering/README.md), then reconcile any commits newer than the evidence snapshot before starting work.
