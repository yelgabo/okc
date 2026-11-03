# Status and next research: distinguish a new mechanism from another local variation

Date: November 2, 2026. Requested by the user: “Status? Deep research into what’s next.” Initial repository evidence pin: `f3a93a2d96d42de3988371425d2cebb200e9bd85`; publication review includes `be8745778ebe145e66db2d1dea0194045ccad661` and its PR cleanup API-contract correction. This is a research/steering report, not a new implementation, H200 run, worker allocation, submission permission, or claim of live CLI access. Existing captain/worker ownership and bounded allocations remain in force. No subagents were launched.

## Decision

There is no new performance-qualified winner. Preserve public layer `607eb52bdaf24160` and stop re-running the exact G224, timed-SiLU, short serial, and subthreshold RMW variants. The strongest currently concrete different mechanism is **hardware-compressible backing for the actual long-load raw BF16 PR weights**. Finish the already allocated ownership repair or record its specific STOP, then make a finite capability-and-benefit decision. Do not confuse allocator/source correctness milestones with a speed result, and do not make a PR-only experiment carry the entire top-100 objective.

The second, higher-risk research direction is **output-owned low-token expert computation with an explicit per-expert normalization boundary**. It is a design discriminator, not a ready candidate or a newly registered architecture family. It challenges the work ownership of T1/T8, rather than repeating the larger-token staged pair or G224. Its feasibility depends on preserving Muse's normalization and repaying any additional boundaries and weight reads. Do not consume a judge slot merely because a Blackwell blog reports a speedup.

## 1. Verified status and what changed since the previous answer

The latest authenticated snapshot inspected is the [13:26 UTC / 05:26 Pacific monitor](../../archive/docs/steering/2026-11-02-serial-terminal-monitor.md): overall and layer rank **351**, public layer **2.1261x**, layer rank-100 point **2.2646x**, one private layer check and five scored slots left, no live job in the returned inventory. These are timestamped repository observations, not new queries made for this report. Overall is separately queried and exposes ranks, not a numerical formula we may invent.

The retained [release manifest](../release-manifest.md) identifies these public results:

| Problem | Public job | Recorded best |
|---|---|---:|
| Layer | `607eb52bdaf24160` | 2.1260826764x |
| Projection | `9eba96894ee1402e` | 1.5278064830x |
| Attention | `8914751e3b9f468b` | 2.6042902049x |
| MoE, historical | `abe7651dbaa44458` | 2.2157115620x |

The MoE number remains historical, not a source-cleared recommendation for another submission: the separate launch-progress/watchdog finding remains unresolved for that exact source. This report has not freshly verified the other three problem ranks. Prefill remains excluded.

Four new ordinary measurements matter more than the count of completed source audits:

| Experiment | Owning private job | Result | Disposition |
|---|---|---:|---|
| G224 chunk-phased expert design | `e1271ad5e5e34b24` | 1.9746646855x; official 13/12/12 and 56 startup checks passed | Exact candidate parked; large-token timings supply no promising signal |
| Timed full-domain SiLU table | `dc62cff972934f54` | 2.073288676x; official correctness passed | Exact treatment parked; no automatic retry |
| Exact unchanged public incumbent | `cc474883117947c0` | 2.1092013572x; official 13/12/12 passed | Fresh control/profile, not a new optimized release |
| Serial BF16 pooler then FP32 router at T1/T8 | `e4c8f00532b74d15` | 2.0930728523x; official 13/12/12 passed | Intended path ran; exact policy parked |

Sources: [G224 terminal](2026-11-02-captain-g224-terminal-decision.md), [timed SiLU terminal](2026-11-02-captain-timed-silu-terminal-decision.md), [fresh incumbent](2026-11-02-captain-fresh-incumbent-profile.md), [serial terminal](2026-11-02-captain-serial-terminal-decision.md). These are actual numerical completions, not all harness failures. Separate jobs have different inputs, routes, preparations and reference timing; they support allocation decisions without proving each underlying causal cost.

### Why independent headline scores are especially misleading now

The unchanged incumbent's fresh own-latency aggregate improved by a factor **1.0197593** relative to historical 607, while its official score fell by factor **0.9920599**. Its SGLang baseline moved by **0.9728373**; the product explains the score movement. The fresh serial run's six own-load times all rose relative to that fresh control, including unchanged long loads. Therefore neither a 1% point-score change nor a 1% cross-job own-latency change proves a treatment effect. The [fresh report](2026-11-02-captain-fresh-incumbent-profile.md) already reconstructs this correctly.

The fresh profile still locates substantial work in expert computation and full-context attention. For example, V32 sliding has a 201.7-us expert activity; D32 full has 171.4-us attention activity. Those are profiler activities, not exposed or removable budgets. Do not sum them into a latency model without dependency evidence. See the [fresh twelve-cell map](evidence/2026-11-02-captain-fresh-incumbent-profile/hotspots.md).

## 2. What we know about the original performance deficit

One genuine defect in the integration was already found and repaired: activation-slot retirement changed compiled WGMMA scheduling from overlapping groups to full waits. Current-stage retirement restored useful overlap while retaining reader lifetime/publication requirements and won its matched component comparison. It is already in 607 and cannot be counted as future upside. [Compiler and H200 evidence](../../archive/docs/steering/2026-11-01-attention-results-expert-pipeline.md), [native repair inventory](../../docs/steering/2026-11-01-g-native-repair-inventory.md).

The broader G0-versus-607 discrepancy remains incompletely explained. The earlier strongest location was T1/T8, not main-only MFIN or a missing SC6 implementation. Subsequent RMW, serial-front and other negative results narrow particular explanations; they do not prove the whole remaining gap is an unavoidable correctness cost. Equally, unsafe original protocols cannot be promoted simply because their historical tests passed. [Integration inventory](../../archive/docs/steering/2026-11-01-g0-607e-integration-gap.md), [current reference audit](../../archive/docs/steering/2026-11-01-g-reference-gap-audit.md).

A new exact source comparison should identify a changed representation, ownership or scheduling mechanism, not reopen every historical flag. Preserve BF16 operator boundaries, FP32 router semantics, graph lifetime, all required reader publication and progress. Distinguish these requirements from choices about how to implement them.

## 3. Primary-source research: which mechanisms actually transfer?

### 3.1 Hopper inline compression: genuinely distinct from the failed software codecs

NVIDIA documents opt-in, per-allocation inline compression on Hopper. Hardware can reduce traffic for suitable data without changing stored numerical values or adding a software reconstruction loop to our kernel. It does **not** shrink the allocated footprint. The VMM guide requires both a device-support query and inspection of the actual allocation properties: a successful request can still return backing without compression attributes. Neither flag proves a useful compression ratio on our weights. [Hopper guide, 1.4.2.3](https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html), [VMM compressible memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory).

This avoids the specific software decoder-versus-tensor-compute competition that hurt P3 and the sparse-repair candidates. It is not evidence that hardware compression will help dense, nonzero BF16 data. Our current B1 lead is old and unmeasured, not a newly invented technique from this search.

The source audit found a concrete active operand: `st.pr`, **4608 x 8192 BF16 = 75,497,472 bytes**, feeding the selected T128/T256 cuBLASLt PR plans with FP32 output. It contains pooler plus router-hi/router-lo rows. Long QKV/O/unpooler raw tensors are separate potential later operands. Short packed dense and NVFP4 expert streams are not replaced by changing PR. [Exact source/API audit, e3b815f](https://github.com/yelgabo/okc/commit/e3b815fc002fef850fb47f2850c748953bcea7df).

### 3.2 Resource ownership is a real issue, but not proof the feature is unusable

CUDA user objects can attach lifetime to graphs and their executable/clone references. Their destructor cannot directly perform CUDA cleanup. Context-wide synchronization is invalid while it encompasses an active capture. This supports the recorded concern about `cuCtxSynchronize`, not a blanket rejection of all VMM allocation. [CUDA 12.8.1 graph and user-object contracts](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#cuda-user-objects).

The live source work must cover graph references, all eager/copy uses, independent preparations, teardown failures, and non-destructive context ownership. Owner-specific noncaptured completion events are narrower than a context wait but are not a complete proof on their own. The concurrent [cleanup API-contract review](2026-11-02-research-pr-cleanup-api-contract.md) adds scoped helper-thread capture-mode handling and cautions about final primary-context release. Its conditional source route does not establish target ABI, runtime acceptance, actual compression or speed. Do not restart that independent review from scratch or report the old WIP as an observed GPU failure.

### 3.3 Warp decode is a useful alternative axis, not a transferable 1.84x claim

Cursor's first-party description assigns warps to outputs, uses two kernels and accumulates routed expert contributions directly. Its published setting is Blackwell with MXFP8 weights and a different operator. [Warp decode](https://cursor.com/blog/warp-decode).

Muse's [task](../../muse_moe_w4a16/TASK.md) normalizes **each expert output before routing-weighted combination**. In general,

`sum_e g_e * RMSNorm(y_e)` is not `RMSNorm(sum_e g_e * y_e)`.

Consequently, directly folding all experts into one running output before normalization is not a legal port. An adapted design needs an expert-wide sum of squares and its required rounding before combination. This is the key screening constraint, not the shared name “MoE.” It prevents an attractive blog diagram from becoming an incorrect implementation.

The transferable hypothesis is output ownership for a very small token population, not removal of Muse's normalization or a promise that the B200 speedup survives on H200.

### 3.4 MonoMoE validates some of our existing direction, not a missing whole architecture

The H200 MonoMoE paper describes weight-major decomposition, placing tokens on the small MMA dimension, persistent execution and readiness-based overlap. Its evaluation is principally FP8; its reported speedups are against other baselines. Our native register-A WGS and fused tiny paths already share several of those ideas, so merely implementing a “persistent, weight-major kernel” again is not an experiment. [MonoMoE primary paper](https://arxiv.org/html/2609.04244).

Use its operator decomposition and readiness schedule as an exact-code comparison only if a specific non-equivalent mechanism survives the Muse math and our rejection ledger. The paper's throughput model must not become a physical lower bound or a claim that our expert phase is globally memory-bound.

### 3.5 TMA/cp.async and Humming: choose per operand, not by instruction fashion

A recent H100 study characterizes TMA versus `cp.async` by transfer size, descriptor reuse and workload. Humming likewise exposes multiple pipelines and per-tensor movement choices. Their results do not establish that replacing our TMA wholesale is beneficial. [TMA characterization](https://arxiv.org/html/2610.34907), [Humming announcement](https://vllm-project.github.io/2026/10/19/humming.html).

Two important applicability checks: the study's paged-attention corpus does not establish a page-size-one result for Muse; and our short `skinny_ring_ws` source already loads activation fragments into registers on a job change. It is not generally an activation-TMA path waiting to be replaced. Likewise, existing WS helper/compute variants can already separate parts of issuing and epilogue work. Inspect actual selected preprocessor flags before proposing that split as new. [Pinned source](https://github.com/yelgabo/okc/blob/f3a93a2d96d42de3988371425d2cebb200e9bd85/muse_decode_layer/cand/g1b4_stage_release.py), `skinny_ring_ws`, `ws_helper`, and WGS definitions.

### 3.6 Another codec headline is not enough

Cloudflare's lossless Unweight work targets Hopper and reduces storage, but its own report also describes significant throughput overhead in its tested integration. That is evidence that storage reduction and faster inference are different objectives, not a universal rejection of codecs. [First-party report](https://blog.cloudflare.com/unweight-tensor-compression/), [research artifact description](https://research.cloudflare.com/nikulin2026/).

Do not restart our measured global/stage-local sparse-XOR losses or large-token software reconstruction solely because another format stores fewer bits. Hardware compression is interesting precisely because it changes that cost model.

## 4. Next implementation decision: one bounded B1 discriminator

Let the existing worker finish its finite graph-owner/cleanup repair or return a concrete STOP. Do not interrupt it, create another delegate, or start a competing allocator implementation. This report allocates no target run.

After source and host-build requirements are satisfied, the captain should choose a single bounded target question that measures **actual usefulness**, not only whether an API returns success. The design should include:

| Arm | Purpose |
|---|---|
| Existing ordinary prepared PR | Deployment/performance anchor |
| Private VMM PR with compression NONE | Controls mapping, ownership hooks and extra allocation machinery |
| Private VMM PR with verified GENERIC backing | Treatment; same values, layout and Lt algorithm as NONE |
| Independent duplicate preparation(s) | Measures module/allocation/state noise rather than reusing one mutable owner |

This is proposed experiment design, not built or judged code. It matters because comparing the default allocator directly with a VMM treatment can mistake changed placement or preparation for compression. Keep actual Lt plans fixed in the first attribution test. Do not tune placement to manufacture a favorable cold-cache result.

Target gates should record: device support; requested and actual backing properties; copied PR bit identity; correct plan pointers before capture; fresh-input and retained-output checks; original eager/capture/clone/destruction lifetimes; all cleanup errors; complete PR-call and layer times. Unsupported or plain backing is a no-treatment outcome, not a successful compressed arm. No zero-filled synthetic tensor may stand in for the judge's actual PR distribution. Where counters are unavailable, report latency and allocation attributes; do not invent physical compressed bytes or HBM utilization.

First separate code admission from payoff. A small explicitly-owned private capability diagnostic may keep all its mappings alive until its own graphs and eager work finish and then clean up normally. That can establish a negative result without first building an unnecessarily general framework, but it neither proves the production caller lifetime nor authorizes a leak, evaluator hook, or new harness cycle. Reuse the actual current worker integration where feasible; a separate diagnostic needs an explicit captain decision and a materially simpler ownership scope.

If GENERIC backing is unsupported, ordinary VMM already adds more overhead than a credible benefit, or the real PR calls do not improve beyond nulls, stop that B1 scope. If actual PR calls improve but the captured layer does not, report the interaction rather than crediting isolated microseconds. If the complete layer improves, confirm independently, then consider other raw long-load operands under the same proved ownership mechanism. Do not immediately extend to packed FP4, KV caches or mutable activations.

## 5. Quantified opportunity: PR alone is a contribution, not a top-100 plan

For six equally weighted load ratios, with fixed baselines:

`score_factor = product_l(C_old[l] / C_new[l]) ** (1/6)`.

Use the fresh control's weighted times, in microseconds: D1 78.765; D16 152.016; D32 191.898; V1 112.682; V16 261.713; V32 351.080. They are rounded from the owning reconstruction. The following are conditional arithmetic thresholds, not measured available slack:

| Hypothetical latency change | Whole-layer factor |
|---|---:|
| Save about 2.726 us on each of D1 and V1 only | 1.0100 |
| Save about 8.817 us on each of V16 and V32 only | 1.0100 |
| Save 10 us on each of V16 and V32 only | 1.01137 |
| Save about 1.482 us on every load | 1.0100 |

The observed layer cutoff/public ratio is `2.2646 / 2.1260826764 = 1.0651514`. Uniform latency would have to shrink about 6.1166%. If just the two long loads carried that whole gap, each would need a 17.2504% proportional whole-load reduction, or about 51.62 us each under an equal-absolute-saving scenario. This is not a claim that PR contains that much removable time. A 1% layer gain from just those two loads requires a 2.9410% whole-load decrease in each.

Therefore: bank a legitimate small gain, but keep the unfilled 6.5% target budget visible. Do not add isolated component deltas that overlap, count the existing stage-retirement fix twice, or equate a memory-compression ratio to a latency reduction.

## 6. A materially different structural reserve: short output ownership with normalization retained

This is my inference from the operator and external work, not an implemented or demonstrated winner. Scope it to T1 first and T8 only if weight reuse survives. Do not reopen the old two-launch staged 128/256 design under a new name.

A candidate ownership sketch is:

1. Gate/up work is owned by output tiles for each runtime token/local-expert pair, reading original activations directly where possible. Preserve NVFP4 dequantization and the required BF16 gate/up and SwiGLU boundaries.
2. Down-projection work is owned by output rows/tiles for that pair and retains FP32 arithmetic plus required scaled-output rounding. Each producer emits its actual expert output and an FP32 norm partial.
3. A norm/combine step obtains each expert's complete norm before multiplying routing weights and accumulating token outputs. No cross-expert normalization substitution is allowed.

The hypothesis is less shared scheduling/coordination at very low token counts, even if an additional launch is required. The risks are more weight reads across T8 tokens, expensive per-expert partial traffic, underfilled tensor instructions or excessive scalar decode work, and losing the fused design's overlap. The current code already fuses the expert path and uses register activation fragments, so no benefit may be credited for deleting preprocessing that is already absent.

Before implementing: compare this exact scope against the old tiny/legacy paths, count actual routed pairs and bytes, derive the per-expert norm communication, and estimate the complete extra-boundary cost using existing measurements. Include an ordinary multi-launch implementation only as a faithful ownership challenger, not as a claim that fewer software barriers always repay more launches. If the plan cannot plausibly address at least roughly 2.7 us across both short loads, or a corresponding weighted budget across other loads, it should not displace the immediate B1 decision. This is a finite design screen, not permission for another tile/warp sweep.

A possible survivor would preserve existing implementations on all losing shapes. A new branch is not promoted from a theoretical peak or the Cursor headline. Required adoption policy, attribution, numerical/progress reviews and matched complete-call comparison remain.

## 7. Changes to how we spend the remaining effort

**Hard correctness and heuristic resource limits are different.** A real missing publication edge or unsafe resource lifetime blocks promotion. A few spills, one rather than two CTAs per SM, or a longer instruction list are performance evidence to investigate, not universal correctness failures. Our own stage-retirement history demonstrates why static “cleanliness” is not the objective. Preserve preregistered experiment decisions; do not retroactively rescue a failed gate.

**One fixed 1% hurdle is not a physical floor.** It is an allocation choice. For future work, a reproducible sub-1% improvement can be worthwhile if inexpensive to integrate and confirm. This does not authorize weakening current frozen gates or reopening all their losers. Evaluate benefit per engineering/measurement cost against the actual score gap.

**Do not use absence of a big proven gain as a reason for endless readiness audits.** For genuinely new mechanisms, feasibility and an explicit conditional budget justify a finite exploratory decision; they cannot prove speed before timing. Conversely, CPU mock count, documentation volume and passing compiler checks must not be reported as performance progress.

**Keep measurements decisive and small.** Avoid another large four-complete-layer artifact or invasive observer when a fixed-policy component question plus the existing whole-call path can decide the mechanism. Any simplification must retain independent mutable state, correct source/job ownership and meaningful held-out inputs. Existing live worker ownership and the final-check reservation are unchanged by this recommendation.

## 8. Ordered next outputs

| Order | Deliverable | Continue / stop |
|---|---|---|
| 1 | Existing B1 owner returns one source/host-build-safe actual PR integration or specific STOP, incorporating the cleanup API correction | No extra readiness loop or duplicate worker |
| 2 | Captain chooses the bounded real-PR NONE/GENERIC/default experiment only when it can decide useful compression and complete-call benefit within available quota | Unsupported, no useful benefit, invalid instrument or lifetime failure stops the exact route |
| 3 | In source-only parallel with existing ownership, screen short output ownership including expert norm and true added traffic | Reject a duplicate old design or an opportunity too small to justify implementation |
| 4 | Confirm a survivor against the exact incumbent and integrate only winning shapes | Matched whole-score evidence, retained gates and attribution; no automatic public submission |
| 5 | Refresh official boards and exact release pins at the next promotion decision | Dated ranks, public best and fresh controlled effect reported separately |

Expected next user-facing progress should be a measured decision about code to keep, not another count of audits. No measured portfolio presently fills the top-100 gap; this is uncertainty, not a statement that the goal is impossible.

## Research provenance and limits

Primary web sources were read for this task: NVIDIA Hopper 12.8.1 tuning guide; current VMM compressible-memory documentation; CUDA 12.8.1 graph/user-object contracts; MonoMoE HTML paper; Cursor's warp-decode implementation description; the workload-grounded TMA characterization; Humming's first-party announcement; Cloudflare's Unweight reports. Claims above are explicitly separated from repository results and proposed designs. No paper's published speedup is assigned to OKC.

Searches included H200/Hopper NVFP4 MoE decode, low-token warp-specialized designs, hardware-compressible BF16 allocation, and current inference-kernel primary reports. Existing fresh repo discovery already found no stronger eligible complete layer in its stated scope; this research did not independently reproduce that entire survey or claim exhaustive absence. It inspected mechanisms instead of repeating the same leaderboard handles. No exact top-ranked private source or per-case competitor receipt was acquired.

No fresh official CLI/queue query, remote H200 benchmark, target compiler run, kernel modification, tolerance change, source adoption or submission was performed by this report. The latest added `be87457` review refines cleanup semantics but reports no performance result. The current B1 implementation is not cleared merely by this report's recommendation. Existing public607 remains the rollback and recorded best layer.
