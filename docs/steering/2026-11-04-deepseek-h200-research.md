# DeepSeek article → current H200 bottlenecks

November 4, 2026. Research only; no native build, GPU job, owner polling, default change, timer change or release action.

## Decision

The supplied Baseten article identifies the right three layers of inference optimization—fusion, Programmatic Dependent Launch (PDL), and host-loop minimization—but those are not missing technologies in OKC. The best-supported current opportunities are specific dependency and placement problems inside an already fused, graph-replayed, PDL-enabled implementation. Do not begin another generic MoE rewrite or a second cached-K/V prefetch experiment because of this article.

Retain the existing attention-d16 merge-contention investigation and MoE-T1 first-weight-issue investigation. A small, distinct reserve is **QSTAGE-EXACT**: preserve the current FP32→FP16→FP8 query conversions while removing the intermediate FP16 shared-memory materialization. This is an implementation hypothesis, not a measured win, a new worker assignment, or a reopening of the closed attention-v1 lane. Its score leverage is small. Nothing in this review establishes a new multi-percent breakthrough.

## 1. Version and project orientation

The inspected repository snapshot is `yelgabo/okc@39ae77cf42b1c8ee8f3d64539c082ac8de02f710`, not the user's local working tree. Earlier reads of `ly_v1un` are an intermediate baseline; the latest inspected engineering/draw base is:

- `muse_decode_layer/cand/ly_stk3.py`, Git blob `7e92d0d202fc1de208324a72dcd05260c2284200`.
- Owning ledger SHA256: `a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6`.
- Private check `4a2bf53969b44e2f`: VALID 13/13, all T enabled, five PDL guards pass, 10,109.93 us, 2.2374x. This is the owner's archived check, not a new measurement or public-row attribution by Research.
- The plan's dated public row is `ly_rfx/caed0290397b44a7`, 10,146.4 us, 2.2294x; dated rank-100 bar 2.2973x. These are repository snapshots, not a fresh leaderboard poll.

The November 4 top-100 plan supersedes short-load-only engineering scope: all sliding cells can be considered by weighted value. Existing owners retain execution and public authority. The older final-day table is not an instruction to repeat terminal General/MoE/BF16 experiments. No routing substitution, precision shortcut, fence removal, or output caching is justified by numerical tolerance.

Sources: [top-100 plan][plan], [latest inspected ledger][ledger], [selected file][stk].

## 2. What the supplied article supports—and what it does not

The supplied text, “How to make flash models even faster: optimizing DeepSeek V4.1 Flash 1017,” describes three mechanisms. It does not provide the optimized kernel implementations; it says related open-source work is planned. Its 2.7 ms router saving and 30% mHC speedup are author-reported results at different scopes, not transferable Muse-layer measurements. Its model-quality, pricing and throughput claims were not needed or independently validated in this pass.

### Kernel fusion

The transferable question is whether a producer/consumer boundary unnecessarily materializes or transforms data. Our attention already combines Q/K processing, rotary position processing, cache updates, attention and output work; our expert engine already contains both expert GEMMs and their intervening/final operations. Pooler/router and residual-boundary fusion also exist. “Fuse MoE” is not an implementation proposal here.

Two qualifications matter. A global-memory write/read is not necessarily a full DRAM round trip; cache may service it. Conversely, a single fused kernel can still exchange cross-CTA intermediates through global memory. Fewer kernel names do not prove fewer exposed transfers. The earlier full-K, norm, finalizer and whole-layer-fusion losses are exact counterexamples to assuming removed work is free speed.

### PDL

This is the article's strongest transferable theme, but our chain already uses it. The active wrapper selects guarded PDL at T=1/8/16/32, and recent wins came from fixing specific edges and input-cache behavior, not discovering the API. The top-of-file `PDL=False` is not sufficient to determine the final executed path; later wrappers and factories change it.

NVIDIA's target-era documentation says PDL allows early secondary execution, requires synchronization before consuming dependent results, and is opportunistic: concurrency is not guaranteed. Resource fit and shared-bandwidth competition remain costs. The article's newer-architecture discussion must not be treated as H200 fine-grained block-level publication permission. Use the actual sm_90a/CUDA-13 contract. [PDL documentation][pdl]

### Host-loop minimization

The judge evaluates CUDA-graph replays, and the local comparison harness captures/replays the complete callable with fresh inputs and a cold-cache step. Python dispatch and allocation during construction/capture must be distinguished from actual replayed device work. `torch.empty` does not imply a replayed zero-fill; `torch.zeros` inside capture can add one. Host simplification can improve prepare time and experimental throughput, but this article does not establish a Python scheduler bottleneck in our scored replay. Breakable prefill graphs address a different serving workload. [Local harness and parity][box]

mHC is a different model operator, not a drop-in replacement for Muse's required residual computation. Its general lesson—co-design the dataflow and execution schedule—transfers; its architecture and headline gain do not.

## 3. What the current measurements actually favor

All numbers in this section are owning-receipt results recorded in the pinned ledger, not Research measurements. Box comparisons are same-session comparisons, not judge-equivalent absolute timing. [Ledger][ledger]

| Mechanism | Recorded result | What it teaches |
|---|---|---|
| T1 unpooler input loads through L1 after the required wait (`ly_xca`) | Decode-1 sliding/full -0.67/-0.65 us, 24/24 paired wins | Target an identified activation-landing delay, not all memory accesses. |
| T8 unpooler PDL edge (`ly_v1un`) | Verify-1 -1.43/-1.40 us, 54/54 paired wins | An unenabled dependency edge can matter more than mainloop tuning. Already integrated. |
| T16/T32 unpooler PDL edges (`ly_d16un`) | Decode-16 -1.76/-2.06 us; decode-32 -0.97/-1.08 us | Expand only an actually compatible dependency mechanism; preserve its input wait/proxy fence. Already integrated. |
| T1 QKV narrow existing sk2 implementation (`ly_q1n`) | Decode-1 -1.32/-0.37 us; other cells within thresholds | Escape-warp/ring ownership was more relevant than nominal tensor-core throughput. Already integrated. |
| Decode-16/32 attention placement (`ly_a16o2`) | Decode-32 sliding -1.19 us; decode-16 -0.21 us | Reducing blocks per SM spread work onto more SMs and improved the endpoint. Already in stk3. |
| T1 register-held unpooler slice, early overlap (`ly_rsl`) | Decode-1 +1.09/+1.46 us | All 128 slice CTAs could overlap and have codes ready early, yet DRAM contention and slower per-CTA math made the complete call worse. Closed. |
| Verify-1 cached K/V before PDL wait (`ly_apw`) | +0.31 us, null-to-loss | Legal independent work is not automatically exposed work. K/V arrival was not the limiting dependency. Closed. |
| Verify-1 per-split merge buffers (`ly_sngw2`) | +1.18 us; finalization 0.8→2.4 us | Less atomic contention can lose to the extra final merge. Closed. |

For T8, the inspected v1un patch also documents a MoE/slice shared-memory incompatibility: their combined dynamic allocation exceeds an SM's capacity. PDL can overlap retiring CTAs without those two CTAs sharing an SM. Do not generalize this into a new register-slice proposal: the separate T1 register-slice trial explicitly achieved residency and still lost. Placement feasibility and endpoint benefit are different claims. [v1un patch][v1patch]

### Ranked existing targets, not duplicate assignments

**Attention-d16: bounded merge contention.** The latest selected decode-16/32 sliding kernel is `attn_kernel<1,false,1>`. The owner found ticket wait scaling with contributors to a 4 KB region: 1.46 us at 16 CTAs, 0.93 us at eight. Its two-copy-by-split-parity `ly_a16nc2` is already built and owned; it was not yet measured in the inspected checkpoint. This is more concrete than “make attention more bandwidth efficient.” The 0.53 us difference is a measured wait difference, not promised complete-call savings. Extra buffer reduction and read traffic must be charged. Leave the owner's implementation/check untouched.

**MoE-T1: first useful weight issue.** The `ly_x1e` early-activation trial was null (+0.19 us sliding). New stamps instead put FAST1's first weight issue roughly 1.1 us after CTA entry, behind routing work. The existing MoE-T1 owner is already investigating this. Distinguish address-independent setup from the minimum route information needed to name the correct expert weights; do not prefetch guessed experts or omit route work. The 1.1 us is an event interval, not 1.1 us guaranteed removable from the complete call.

**Layer b1/b2:** an existing owner is studying decode-16/32 boundaries. Its floor-distance estimate is a prioritization clue, not a certified 6–7 us recoverable budget. Preserve this lane rather than starting another fusion owner.

## 4. One new source-grounded reserve: QSTAGE-EXACT

### Exact scope and distinction

Target only the current verify-1 sliding attention route: B=1, q_len=8, the supported page/window/layout conditions, selected `attn_kernel<8,false,15>` and its current 33 splits. Preserve the incumbent for every other path, neighbor and fallback. No splits, thread count, routing, tile geometry, ring depth, cache policy or PDL placement sweep.

The prior `ly_aqd` experiment uses direct FP32→E4M3 query conversion and can use a different query-lane grouping. Its approximately 0.42 us signal was below its owning keep rule and is not an accepted gain. The reserve is different: retain the baseline eight-lane reduction and **both existing quantization steps**, changing only where the intermediate representation lives.

### Actual selected computation

In `ly_stk3.py`:

- Lines 240–286 establish the non-flat L8 configuration with 256 threads, no separate flat producer warpgroup, and disabled two-stream mode.
- Lines 390–460 define `pack_h2` using `__floats2half2_rn`.
- Lines 615–642 define `pack_e4m3x4`, converting two raw half pairs with `__nv_cvt_halfraw2_to_fp8x2` and saturation.
- Lines 1650–1825 contain the original eight-element-per-thread query normalization/rotary loop and its FP16 shared-memory store.
- Lines 1975–2070 reload that FP16 representation, convert it to FP8, and store the final swizzled `S.q8s` matrix, followed by the generic-to-async proxy fence.

Baseline dataflow:

```
BF16 Q input -> unchanged FP32 normalization/RoPE -> pack_h2
            -> FP16 shared store -> FP16 shared reload
            -> pack_e4m3x4 -> FP8 shared matrix -> WGMMA
```

Reserve dataflow:

```
BF16 Q input -> unchanged FP32 normalization/RoPE -> pack_h2 in registers
            -> pack_e4m3x4 -> same FP8 shared matrix -> WGMMA
```

Do not set the old QDIRECT switch. Add an isolated exact-staging specialization and bypass only the obsolete FP16 materialization/reload branch on that specialization. Keep `qsc`, the original FP32 expressions, the eight-lane sum order, finite/nonfinite handling, cache updates and output/gate operations.

A scalar example explains why retaining the half conversion matters: FP32 1.0626000166 rounds to FP16 1.0625; nearest-even E4M3 then rounds to 1.0. Direct E4M3 conversion rounds the original FP32 value to 1.125. This is a CPU rounding witness, not a CUDA tolerance failure.

### What was checked in this pass

A CPU enumeration compares the original consumer-side row/column swizzle with the proposed producer-side stores. All **8,192 logical FP8 bytes** map identically to offsets 0–8191, with exactly one proposed writer per destination. Mapping SHA256: `939f02946ba255125d8e66c7e6d7c5bb65e87d1ca052c628c4d0f5d0d833ce78`.

Static traffic accounting removes 16 KiB of FP16 shared writes plus 16 KiB of shared reads per CTA: 32 KiB/CTA, or 1,081,344 bytes for 33 splits. These are **shared-memory bytes, not HBM savings and not an exposed-time estimate**. No native compilation, SASS comparison or GPU execution was performed.

### Competing costs and opportunity

The owner measures the complete query/RoPE pass at about 1.0–1.1 us, with only 0.11 us in the RoPE table itself. This candidate can remove only a portion of that pass plus its later restaging; it retains all numerical conversion work. It may lose if moving the conversions earlier lengthens a dependency chain, increases live registers, changes store scheduling, or shifts cost into another critical region.

The earlier direct-query signal justifies one bounded implementation hypothesis; it does not forecast this exact candidate's result. A **0.4 us net sliding-cell improvement** is a useful decision point under the latest plan, worth approximately **0.049% of the six-load layer score** at the dated 101.2 us cell time. No credible 2 us claim or multi-percent score claim follows. This reserve should not displace a better measured active-owner candidate.

### Proof and lifetime obligations

Preserve the existing FP16-rounding and half-to-FP8 operations in emitted code; compare the final FP8 query tile byte-for-byte to the base, not merely the final norm. Do not borrow QDIRECT's four-lane reduction layout. Store to the existing distinct `q8s` region without aliasing raw Q, K/V staging or its later transpose workspace. Retain a writer-side `fence.proxy.async.shared::cta` and the existing inter-thread synchronization before any WGMMA consumer. Keep new-K/V cache stores, denominator/softmax behavior, PDL waits/triggers, merge publication, progress and graph scratch lifetime unchanged. No partial output may escape its existing visibility contract.

The CPU map check proves address coverage only. Resource usage, target code generation, final factory dispatch and the native ordering argument must be checked during a build. A correct bounded implementation may be built before its speed is known.

### Owner, bounded discriminator and stop

This is a reserve for the **existing attention implementation owner/final-day lead**, not an instruction to restart the closed attention-v1 chat. Suggested engineering cap if selected: two hours, one fixed exact-staging specialization, no geometry variants. No allocation follows from this report.

The smallest meaningful comparison is the full layer callable from exact `ly_stk3` versus the one exact-staging variant. Prepare independent mutable states; preserve every other selection and PDL edge. Check the internal FP8 tile separately, then all outputs, routing, cache writes and repeated graph execution. Use matched fresh normal/adversarial inputs, cold replay, duplicate controls, all 12 layer cells, and at least 30 paired repetitions per cell. Inspect conversion instructions, shared accesses and resources without treating a static count as speed proof.

Stop on an unclosed ordering/lifetime/progress obligation, tile mismatch, native numerical failure, or failure to clear the current complete-cell/noise gate in at most two independent paired sessions. Do not rescue with altered precision, a new split count, a new ring, a different warp count, an input mask or a selected-seed claim. Judge/public execution remains the lead's separate decision.

## 5. Additional H200 lessons from primary implementations

**Independent setup before dependent work.** DeepGEMM at `5081f8513af57e61696de4f4f04b0e707ffb503e`, `sm90_fp8_gemm_1d1d.cuh` lines 25–200, prefetches TMA descriptors and initializes its barriers before `cudaGridDependencySynchronize`, then starts dependency-sensitive scheduling. It also allocates different producer/consumer register budgets. This is a concrete source example of ordering setup by dependency, useful to the already-owned MoE startup analysis. Its FP8/per-128-channel arithmetic is not a drop-in Muse NVFP4/group-16 expert implementation. [Pinned source][deepgemm]

**Overlap different resource classes, not merely adjacent kernels.** FlashAttention-3 overlaps tensor-core matrix work with softmax and data movement. The principle transfers to Hopper; its headline large-attention results do not predict one-tile decode behavior. Our source already has specialized producers, FP8 WGMMA and pipelined PV handling. Generic “add warp specialization” is stale, and arbitrary ping-pong barriers can add serialization. [FA3 author's explanation][fa3]

**Placement can matter more than maximum occupancy.** Hopper's register/shared-memory limits determine which blocks can coexist; cluster occupancy must be checked explicitly. In our current layer, the host-only attention reservation deliberately reduced blocks per SM and improved completion. Conversely, the resident register-slice design lost. Optimize complete-chain throughput under contention, not occupancy as a standalone score. [Hopper guide][hopper]

**CUDA-13 shared-memory spilling is not a free compiler switch here.** PTX 9.0 disallows `enable_smem_spilling` for kernels using dynamic shared memory or `setmaxnreg`. The inspected attention source uses dynamic shared storage; flat configurations additionally repartition registers. A different static-storage design would require its own cost/progress analysis, not a pragma added to the current kernels. [Target-era PTX][ptx]

## 6. Score realism and next decision

For a cell with normalized weight w, reducing time t by delta multiplies the score by `(t/(t-delta))**w`. With the plan's 46:16 sliding/full weighting across six loads, each sliding cell has w=46/372. The following are exact conditional calculations from the dated table, not forecasts:

| Hypothetical endpoint improvement | Layer-score improvement |
|---|---:|
| Verify-1 sliding -0.4 us | 0.049% |
| Decode-1 and verify-1 sliding -1 us each | 0.306% |
| Decode-16 and decode-32 sliding -5 us each | 0.926% |
| Every sliding cell 5% faster | 3.879% |

The dated public gap is about 3.05% in speedup ratio; the checked stk3 point reduces that to about 2.68% against the same dated bar, but the bar can move and checks are not public rows. The table's ideal-floor gaps and the earlier overlapping activity totals are not an exposed savings budget. Repeated favorable draws are statistical selection, not algorithmic speedup.

The implementation decision is therefore: prioritize the already-owned, measured dependency/merge/placement opportunities; retain QSTAGE-EXACT as a low-cost, numerically conservative reserve; do not claim the article supplies an unimplemented 3% optimization. The incumbent is not proved optimal, but new proposals must explain the complete dependency path and the costs they introduce.

## Evidence and limits

The supplied article was read from the user's `Pasted text(1).txt` (53 indexed lines); it was not replaced by a guessed article from its title. Primary public documentation and the pinned DeepGEMM implementation were inspected independently. Repository facts above are tied to main39; local/uncommitted changes, live job status, full leaderboard refresh, compiled candidate performance and current-owner pickup are not established. Other owners' IDs were read only in their archived ledger, never polled.

Reproducible CPU layout/score results: `qstage_mapping_check.py` and `qstage_mapping_result.json` in the delivered research package. Existing historical reports and failed variants are preserved. This document changes neither task ownership nor compute/public permission.

[plan]: https://github.com/yelgabo/okc/blob/39ae77cf42b1c8ee8f3d64539c082ac8de02f710/docs/steering/2026-11-04-top100-push.md
[ledger]: https://github.com/yelgabo/okc/blob/39ae77cf42b1c8ee8f3d64539c082ac8de02f710/docs/steering/2026-11-02-final-day-directive.md#ledger
[stk]: https://github.com/yelgabo/okc/blob/39ae77cf42b1c8ee8f3d64539c082ac8de02f710/muse_decode_layer/cand/ly_stk3.py
[v1patch]: https://github.com/yelgabo/okc/blob/39ae77cf42b1c8ee8f3d64539c082ac8de02f710/tools/dl_ly_v1un_patch.py
[box]: https://github.com/yelgabo/okc/blob/39ae77cf42b1c8ee8f3d64539c082ac8de02f710/docs/box.md
[pdl]: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization
[ptx]: https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#pragma-strings-enable-smem-spilling
[hopper]: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
[deepgemm]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh#L25-L200
[fa3]: https://tridao.me/blog/2024/flash3/

---

## Addendum — November 4, 2026: barrier-removal results and current head

**Reorientation through `f370e3aa84114ecef5d6e71a98c1c673996d3d15` (14:06 PT ledger entry).** The source/result inspection started at `a88fa79fe225d3d3b5ee12668faa929efd7fd4ca`; the subsequent head adds the existing DF/YBF lane. Everything above remains the original `39ae77cf` snapshot. This appendix supersedes its current-status recommendations without rewriting its evidence, experiments or conclusions as though the newer results had been known then.

### A. The selected base has advanced; the two barrier results have different statuses

| Implementation | Changed selected path | Owning complete-layer box comparison | Latest archived judge checks / disposition |
|---|---|---|---|
| `ly_stk4` = `ly_stk3` + `ly_gb4` | B4, 16 < T <= 32; scored decode-32 sliding/full | Against stk3: -1.75 / -1.67 us, 24/24 paired wins in each cell; other cells within thresholds | VALID 13/13 at 10,046.14 and 10,070.38 us. **Current engineering/draw base.** |
| `ly_stk5` = `ly_stk4` + `ly_gbm` | MAIN/RFIN; scored verify-16/verify-32 (T128/T256) | Against stk4: verify-16 -1.03 / -1.04 us; verify-32 -1.59 / -1.40 us, respectively sliding/full | VALID 13/13 at 10,111.61 and 10,101.49 us. **Recorded alternative, not the selected draw base.** |
| `ly_gb5` | MID T16, separate per-job completion implementation | Against stk4: decode-16 sliding +0.11 us, full -0.01 us | Null under its bar; closed and not stacked. |

The two-draw arithmetic means are approximately **10,058.26 us for stk4** and **10,106.55 us for stk5**. These are cross-job score observations, not a matched causal estimate that MAIN adds 48 us. The latest ledger says much of the difference appears in unchanged decode-32 code; the MAIN-changed cells are slightly ahead at verify-16 and approximately even at verify-32. The lead nevertheless retained stk4 on its observed record. Keep both the MAIN box gain and the inconclusive judge attribution, rather than calling MAIN either a confirmed official improvement or a proven regression. [Latest ledger][head-ledger]

Exact source identities: stk4 SHA256 `a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e`; stk5 SHA256 `b89d23a6a00d5d00a48821e419135727388dcad1349e4924e9a37591f9f74654`. Owning check IDs are stk4 `4f99c1ec20f54f0f` / `d947533cf0c540a2` and stk5 `1e0d522594094eef` / `c3112d526fe0489d`. These were read from archived repository evidence, not polled. The selected draw base is not automatically the public row; this pass made no leaderboard request or submission.

### B. What “barrier removal” actually means here

**The successful change narrows the dependency, not the memory-ordering guarantee.** The old path waited for all CTAs to finish expert work before finalization. With MFIN enabled, each final item acquires the completion counters of its token's local experts; unrelated experts do not need to finish first. Expert arithmetic, the final sum and the required publication remain. The useful question is now which data a final item actually needs, rather than how cheaply to implement the same all-grid wait.

The inspected source supports that distinction:

- The pre-finalization grid barrier is conditionally excluded by `OKC_MFIN && OKC_WGS && (OKC_MFQ & 4)`. B4's existing column-parallel finalizer polls its required local experts with `ld_acquire`, performs `CTA_SYNC`, then reads the output/norm partials. Its fixed-order reduction and final BF16 conversion are unchanged. [B4 finalization][head-stk4-final]
- `wgs_flush` executes a 128-thread warpgroup barrier before one leader issues `red.release.gpu.global.add.s32` for a kind-4 expert completion. This is not the old proposed WS relaxed-flag relay, and the earlier unproved WS publication argument is not repaired or vindicated by these results. [Publication source][head-publication]
- The readiness polls no longer fall through after four million unsuccessful iterations. This removes the unsafe “continue without readiness” path. It does not by itself prove progress: counter reachability, all-contributor coverage, resident scheduling and generation/reset lifetime remain necessary. Passing outputs are not a replacement for those obligations.

B4 obtained a complete-call win from this dependency change; MID did not. The MID owner's hypothesis is that each token still waits for a last-finishing phase-B job, leaving the end spread exposed. That is an explanation to test, not an established family ceiling. A grid-barrier interval is not an additive removable-time budget. [MID receipt][head-mid]

### C. MAIN's reopening corrected an actual source premise

The earlier statement that RFIN finishers never release MFIN completion credits was wrong. In the current selected source, a non-finisher split share writes fixup partials and returns; the stage-zero finisher acquires those shares, combines their partials, and **falls through into the whole-unit epilogue**. That epilogue sets `WgsPend{4,...}`, and `wgs_flush` publishes the completion credit. The final pending release is also flushed at phase exit. [RFIN control flow][head-rfin]

The owning source argument counts four warpgroup credits per phase-B unit and sixteen units per job: **64 credits per job**, matching the finalizer target times that expert's job count. The MAIN gate could therefore retain MFIN without inventing a missing finisher protocol. The owner additionally tested **5,800 graph replays** across M=33/48/97/128/200/255/256, varied routing, exact counter targets, opposite-parity zeroing, alternating parity and bitwise equality to stk4. This is strong recorded implementation evidence, not an independent exhaustive ordering/progress proof by this appendix. [MAIN owning proof and stress][head-main]

The practical research correction is important: **audit the exact source premise behind a closure before ruling out a compatible mechanism.** This was new control-flow evidence, not a reason to repeat every previously losing barrier/flag treatment. The exact WS flags, full-K, norm and two-launch losses retain their existing scope.

### D. Explicit corrections to the earlier recommendations

| Earlier statement in this report | Newer evidence and current disposition |
|---|---|
| Prioritize attention-d16's two-copy merge candidate `ly_a16nc2` | The complete-layer comparison finished null: decode-16 sliding +0.03 us, 12/24 faster, despite an earlier stamped O endpoint. The lane closed. Do not continue recommending it as pending. |
| Prioritize T1's first weight issue “behind routing” | The subsequent source/timing reconciliation shows TINY already issues its first two copies before `route()`. `ly_f1f` advanced issuance by about 0.15 us but was null (-0.24 us sliding, 16/24); issue-to-first-slot-full was about 1.93 us. The lane closed. The earlier attribution is superseded, not silently retained. |
| Layer b1/b2 is an ongoing measured target | The read-ahead arm was null, and T32 cluster glue lost +1.04 us. This lane closed; its old floor-distance estimate is not remaining recoverable time. |
| MAIN cannot use completion counters because RFIN does not publish | Corrected by the actual finisher fall-through and counter accounting above. A valid MAIN candidate exists; the lead retains it as an alternative rather than the current base. |
| `ly_stk3` is the current comparison base | Use `ly_stk4` for newly selected comparisons. Preserve `ly_stk5` and its owning receipts as a separate alternative. |
| QSTAGE-EXACT is a small reserve | Still source-only, unbuilt and unmeasured in the inspected evidence. No new pickup is established. It is not promoted by the unrelated barrier results; any future comparison must use the then-current base. |

Additional completed work: the own-GEMM/PDL alternative at T128 was not a material sliding-cell keeper; broad launch/staging revival closed without a new selected candidate. These newer terminal entries supersede the older “in flight” descriptions in the original report. [Current closures][head-ledger]

### E. Reoriented recommendation and latest ownership

The strongest new lesson is **consumer-specific readiness with complete contributor accounting**, not generic barrier deletion. B4 already demonstrates that mechanism in a complete measured call, while MAIN's source correction shows why exact control-flow inspection can change an earlier integration decision. Preserve the existing release/acquire chain, precision boundaries and replay/reset behavior when assessing any related path.

At the final inspected head, the latest user-authorized lane is **`moe-dfybf`**, opened at 14:06 PT: research DF/YBF features still off, build on stk4 only for a few-line justified repair with a plausible >=0.4 us sliding-cell opportunity, stop at 17:30 PT, **no judge use by that lane**. This is an existing owner's assignment. It supersedes any blanket claim that every implementation lane is now closed; it is not a new assignment or allocation from Research. [Head delta][head-delta]

For this report, the immediate ordering is: retain stk4 and the lead's current release/draw decision; preserve stk5's distinct evidence; leave DF/YBF with its already-opened owner; keep QSTAGE-EXACT as a lower-leverage reserve only. Do not restart the completed attention merge, MoE prologue, MID barrier or generic knob searches.

The plan's approximately 2.1% remaining gap uses its dated top-100 reference and stk4's best check. It is not a live rank statement, and neither the full 63.79 us difference from stk3's single check nor the old barrier idle spans can be credited as a causal algorithmic saving. Repeated draws and implementation gains remain separate. [Updated plan][head-plan]

### F. Preservation and limits

This is an append-only update to the same report. The original **20,934-byte UTF-8 prefix** is retained, Git blob `129595fd0d5e13851fc71c174281808a07029184`, SHA256 `ee6388723089d2edebc7f3d76398d0bccf54fb2ba0b0a4f4c44849dd2115f668`. Its earlier links stay pinned to the original snapshot; the references below bind this appendix to the newer evidence. No earlier result is relabeled retroactively.

This pass read pushed source, current steering and owning archived results. It did not inspect uncommitted local changes, run native compilation or GPU measurements, poll owner jobs, change kernels/defaults, edit owner queues, schedule anything, or take public action.

[head-ledger]: https://github.com/yelgabo/okc/blob/f370e3aa84114ecef5d6e71a98c1c673996d3d15/docs/steering/2026-11-02-final-day-directive.md#ledger
[head-plan]: https://github.com/yelgabo/okc/blob/f370e3aa84114ecef5d6e71a98c1c673996d3d15/docs/steering/2026-11-04-top100-push.md
[head-stk4-final]: https://github.com/yelgabo/okc/blob/f370e3aa84114ecef5d6e71a98c1c673996d3d15/muse_decode_layer/cand/ly_stk4.py#L6505-L6630
[head-publication]: https://github.com/yelgabo/okc/blob/f370e3aa84114ecef5d6e71a98c1c673996d3d15/muse_decode_layer/cand/ly_stk5.py#L5890-L5910
[head-rfin]: https://github.com/yelgabo/okc/blob/f370e3aa84114ecef5d6e71a98c1c673996d3d15/muse_decode_layer/cand/ly_stk5.py#L6079-L6168
[head-main]: https://github.com/yelgabo/okc/blob/f370e3aa84114ecef5d6e71a98c1c673996d3d15/docs/lanes/moe-gbm.md
[head-mid]: https://github.com/yelgabo/okc/blob/f370e3aa84114ecef5d6e71a98c1c673996d3d15/docs/lanes/moe-gb.md
[head-delta]: https://github.com/yelgabo/okc/commit/f370e3aa84114ecef5d6e71a98c1c673996d3d15
