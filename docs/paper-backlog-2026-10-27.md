# Open Kernel Challenge — Paper-Derived Experiment Backlog

**Snapshot:** October 27, 2026, late night Pacific; some repository logs are dated October 28 UTC.  
**Repository:** https://github.com/yelgabo/okc  
**Purpose:** Research candidates for the existing H200 project, not a restart or authorization to submit.  
**Evidence labels:** MEASURED = user/repository measurements; DERIVED = calculations from inspected code; PUBLISHED = source method; HYPOTHESIS = proposed adaptation; GATE = suggested decision threshold, not a predicted gain.

## Executive decision

Keep the in-progress FP32-accumulator-mode/FP32-softmax attention check first. After that, choose the next attention experiment from its measured critical path:

- **Consumer/softmax limited:** A1 bounded-shift rescaling; then A5 inter-warpgroup staggering.
- **Loader/resource limited:** A2 shared-memory lifetime compaction; then A4 row-granular TMA only after an address-layout and instruction-cost audit.
- **Short sliding-window workloads:** A3 in-kernel split merge is a separate opportunity with a measured merge-cost target.

Retain the two already queued MoE experiments, but sharpen them: M1 hot/cold expert tile selection and M2 packed-weight TMA. Do not call them newly discovered ideas. Add B1 hardware-compressible BF16 allocation as a capability-gated architectural probe, not as a promised optimization.

**Important changes from the earlier reading list:** Stream-K is conditional on demonstrated work imbalance, not the default remedy for repeated hot-expert weight reads. FlashDecoding++ has a recomputation fallback; importing just its fixed softmax shift would omit an essential part. SonicMoE's token-rounding router changes real routing and is not an acceptable shortcut here.

No GPU experiments were run for this research. A small CPU-only numerical check of A1's algebra was run; its scope is recorded below.

## 1. Baseline to preserve

The user's latest submitted results are BF16 projections **1.139x**, MoE **1.699x**, attention **1.291x**, and integrated decode **1.390x**. The **1.348x attention result is checked, not submitted**, and uses FP16 Q.K accumulation and half2 softmax. The precision-conservative replacement is still in progress in this snapshot.

The W4A16 integrated-layer port and the score-weighting fix are complete. Do not create tasks to redo either. The whole-layer W4 path passed 13/13 tests, has no shape regressions, and uses approximately 95.5 seconds of the 300-second preparation allowance. Programmatic dependent launch makes individual kernel durations unsuitable for additive latency forecasts.

Measured attention facts driving this backlog:

- Full batch-32 verify: approximately 332–333 us on the checked warp-specialized implementation, versus about 386 us for the FA3 path.
- A cold-L2, shuffled-page, 128-byte-row gather benchmark: approximately 181 us for 655 MB, about 3.6 TB/s. The 64-byte-row version achieves substantially less bandwidth.
- The checked kernel's loader-only diagnostic is around 300 us. Loader issue and value conversion remain substantial; a raw gather ceiling is not a demonstrated full-attention ceiling.
- The current consumer path contains two warpgroups, one per KV head, and four loader warps.
- The repository attributes 16–24% of sliding-verify time to the separate merge launch.

Measured MoE facts:

- GEMM1 at batch 8 reaches approximately 3.25 TB/s, around 85% of the measured comparable streaming bandwidth.
- Removing dequantization mostly changes time by only 2–4%.
- At batch 256, hot experts can receive 256 tokens; current 8–16-token work items repeatedly access their weights, often through L2.
- Persistent cross-item prefetch, alternative weight layouts, streaming addresses, prefetch-hint changes and the cluster finalizer did not establish durable gains.

Project sources: the supplied status update; `docs/runs-attention.md`; `docs/runs-moe.md`; `muse_attention_decode/attn_ws.cu`. The inspected `attn_ws.cu` blob SHA was `a9ea3d14bce3f83d43dd6297a3697f306eade2e8`. This is the checked FP16-softmax implementation, not an assertion about an agent's uncommitted FP32 rewrite.

## 2. Reading map: what was actually useful

These are focused readings of methods, algorithms, relevant figures, limitations and implementation constraints—not claims to have reproduced each paper's experiments.

| Source | Sections examined / useful mechanism | Decision for this project |
|---|---|---|
| [R1] FlashAttention-3 | Sections 3.1–3.3; producer/consumer execution, inter-warpgroup ping-pong, intra-warpgroup overlap, register costs and FP8 layout constraints | Use for A2/A5. Do not equate all forms of pipelining with the already failed paired-tile variant. |
| [R2] FlashAttention-4 | Sections 3.1.3–3.1.4; exponential emulation and delayed rescaling | A1 is portable algorithmically. Polynomial exp is conditional. Blackwell TMEM/2-CTA MMA is not an H200 mechanism. |
| [R3] FlashInfer | Sections 3.2–3.3; scattered KV loads, tile/resource choices and planning | Useful for resource budgets and conditional D1. Its CPU planning must not be hidden outside timing for fresh contest inputs. |
| [R4] Stream-K | Algorithms 4–5, Sections 5.2 and 6; iteration balance versus cache skew and reduction overhead | Defer a wholesale scheduler rewrite. First demonstrate an underfilled tail rather than bandwidth/weight-reuse limitation. |
| [R5] FlashDecoding++ | Section 3 and Figure 6; unified shift with range checking and recomputation | Not simply an unsafe fixed maximum, but the complete mechanism is more complicated than A1. Defer. |
| [R6] ThunderKittens | Section 3.2 and Appendix B.1; worker/pipeline depth and storage/occupancy tradeoffs | Apply its resource-accounting discipline, not a framework rewrite. |
| [R7] LeanAttention | Section IV and Figure 4; uneven work partitioning and associative partial-softmax merging | Supports D1/A3 reasoning. Your existing partial-softmax merge is already this kind of algebra. |
| [R8] SonicMoE | Algorithms 1–3, Sections 4–5 and Appendix B; grouped-GEMM I/O, pipelining and tile utilization | Refines M1/M2. Token-rounding routing and training/backward optimizations do not transfer directly. |
| [R9] FLUTE | Section 3; lookup-oriented quantized GEMM and offline restructuring | Existing fragment repacking addresses much of the transferable opportunity. Do not reopen dequantization without new evidence. |
| [R10] POD-Attention | SM-aware co-scheduling of independent prefill/decode work | Serving-track reserve, not a homogeneous kernel-track optimization. |
| [R11–R14] NVIDIA / Colfax technical references | Hopper resources, TMA, asynchronous memory ordering, instruction targets, compressible allocations | Hardware constraints and the B1 probe. |

Paper headline speedups are **not forecasts for Muse/H200**. The model shapes, precision, page size, hardware and timing methodology differ.

## 3. Backlog index

All gains below remain unmeasured. Suggested checks are experiment budgets, not promises that implementation or validation will fit in that many attempts.

| ID | Candidate | Status | Admission condition / first evidence | Suggested initial GPU budget |
|---|---|---|---|---|
| G0 | Precision-conservative attention control | Already in progress | Establish FP32-mode performance, error and phase timings | Existing planned bench + final check |
| A1 | Bounded-shift FP32 online softmax | New, high priority if math-limited | Rescaling/consumer work is on the critical path | One grouped comparison, then validate winner |
| A2 | Compact staging lifetimes to change occupancy | New, high-value resource investigation | Compile-time budget permits a different residency point | CPU/layout audit first; one grouped comparison |
| A3 | Sliding-verify cluster merge and norm-gate | New for q_len=8; q_len=1 precedent exists | Merge is a material fraction; cluster storage fits | One bounded 2/4-CTA comparison |
| M1 | Larger tiles only for hot experts | Already queued; refined | Actual hot-expert rereads, not merely many tokens | One grouped routing-seed comparison |
| M2 | TMA for contiguous packed W4 weight tiles | Already queued; refined | Net register/issue benefit survives new shared-memory costs | One grouped transport comparison |
| A4 | Row-granular TMA for attention K/V | New, conditional | Feasible layout and command-rate model | One diagnostic comparison before full integration |
| A5 | Stagger the two consumer warpgroups | New, conditional | FP32 consumers compete for the same execution resource | One comparison; no blind paired-tile retry |
| D1 | GPU-resident length-aware scheduling | Conditional / defer | Measured last-wave imbalance on unequal lengths | No check until a tail is demonstrated |
| B1 | Hardware-compressible BF16 weight allocations | New architectural probe | Device and allocation support, admissible allocation path | Capability check piggybacked; one paired probe if supported |
| N1 | Partially emulate FP32 exp2 on FMA units | Conditional numerical experiment | MUFU remains limiting after A1/A5 and rules permit it | CPU error audit first |
| N2 | Native FP8 WGMMA P.V | Held pending precision interpretation | Probability precision is explicitly permissible | No performance check yet |

## 4. G0 — Control and correctness gate

Keep the original submitted attention route as the fallback. Compare the new FP32-mode kernel against FA3 and the checked FP16-mode kernel in the same diagnostic job, but route production experiments from the precision-conservative control.

Record Q.K instruction type, softmax arithmetic type, probability operand type, P.V accumulator mode, register counts, shared memory, spills and barrier policy. Keep FP8 query/cache precision and all BF16 boundaries intact.

**Terminology caveat:** NVIDIA documents that FP8 `wgmma.mma_async` with `.f32` destination currently accumulates internally at precision above half but below single precision [R12]. Describe the variant as **FP32 accumulator/output mode with FP32 softmax**, not an IEEE-FP32 guarantee for every internal tensor-core addition. This does not establish a contest violation; compare the allowed baseline's instruction semantics and obtain an organizer interpretation if necessary.

Also audit the producer/consumer happens-before chain. The current log reports a gain from removing a loader-side async-proxy fence. Passing numerical tests alone is not proof that a fence is unnecessary. Preserve a documented legal ordering between generic shared-memory writes, barrier publication/acquisition and WGMMA reads. This is a review gate, not a claim that the checked implementation has a race.

## 5. A1 — Bounded-shift online softmax, retaining FP32 arithmetic

**Source:** [R2] Section 3.1.4; online-softmax context [R1], shift invariance [R5].  
**Hypothesis:** Avoid repeated output-vector rescaling without going back to half2 softmax or ignoring low-attention keys.

This is different from the prior optimization that merely skips rescaling when the running maximum does not change. Allow the normalization shift to lag a modest amount behind a newly observed maximum.

### Proposed base-2 formulation

Let `x` be logits expressed in log2 units. Maintain a retained shift `c`, denominator `l`, and unnormalized output vector `u`:

```text
l = sum_j 2^(x_j - c)
u = sum_j 2^(x_j - c) * V_j
output = u / l
```

For each tile:

```text
Ignore masked entries when finding tile_max.
If the tile has no valid entries, do not change state.
If state is empty, initialize c = tile_max.
Else if tile_max > c + tau:
    a = exp2(c - tile_max)
    l = a * l
    u = a * u
    c = tile_max
p = exp2(x_tile - c)             # masked entries remain exactly zero
l += sum(p)
u += p @ V_tile
```

Changing `c` and rescaling **both** `l` and `u` preserves their common normalization in real arithmetic. Do not update the shift while leaving either accumulator in the old scale. With this rule, `p <= 2^tau` for valid finite logits. Start with `tau = 0, 2, 4, 8`, not a large guessed range.

This does not skip keys, terms or small probabilities. It changes when an equivalent normalization is performed. Finite-precision behavior still needs official testing; final normalization cannot undo overflow or earlier rounding errors.

### Smallest experiment

Hold tile sizes, load pipeline, query rounding and all input/output types constant. Compare the conservative control with two or three thresholds. Add a rescale counter and phase stamps only in the diagnostic build. Use a warp-consistent rescale decision where needed to avoid divergent collective operations.

Test increasing maxima, sudden outliers, empty split partitions, window-edge masks, 1/64 cache scales, coherent low-attention values and the million-token correctness case. Preserve the legal P operand type and bound its representable range.

**CPU evidence only:** A float64 implementation of the stated recurrence was compared with stable dense softmax on 116 configurations, including 80K and 1,048,576-token inputs, masking, increasing maxima and coherent low-weight values. Maximum observed relative L2 was `8.45e-15` (rounded up). This checks the algebra and basic masking, **not** CUDA synchronization, FP32/FP16 rounding, contest legality or speed. The companion script `okc_bounded_shift_softmax_check.py` reproduces it (`python okc_bounded_shift_softmax_check.py`, with NumPy installed).

**GATE:** Advance only if whole-path latency improves repeatably and numerical margin remains sound. Stop if consumer cycles fall but loaders still determine unchanged runtime. Do not forecast a large gain from rescale counts alone.

## 6. A2 — Shared-memory lifetime compaction, not another deep-stage sweep

**Sources:** resource tradeoffs [R1, R6, R11]; inspected `Smem<NS,DA>` in `attn_ws.cu`.  
**Hypothesis:** A smaller live storage footprint can permit more independent work to reside on an SM and hide loader latency better than a single deep pipeline.

### DERIVED resource budget

For the checked `NS=7, DA=3` implementation:

| Allocation | Bytes |
|---|---:|
| Seven K + raw-FP8-V stages | 114,688 |
| Four converted-FP16-V stages | 65,536 |
| Separate new-token K/V buffers | 24,576 |
| Query buffers | 8,192 |
| Barriers | 112 |
| **Total** | **213,104 = 208.109 KiB** |

The checked alternative stage pairs still remain in the same one-block shared-memory regime: `(6,2)` is about 192.094 KiB; `(8,4)` about 224.125 KiB. Hopper's documented 228 KiB per-SM shared-memory capacity rules out two such blocks [R11]. This explains why the previous 6/8-stage comparison does not test a two-block design.

### Proposed adaptation

Draw a lifetime diagram for K, raw V, converted V, Q, new-token staging and final partials. Give K and V independent release points where their asynchronous consumers genuinely finish. Construct new-token tiles after the main loop into dead staging memory rather than reserving a dedicated 24 KiB region throughout the loop. Check that removing their early staging does not create an unacceptable tail.

An illustrative target is four FP8 stages, two FP16-V stages, Q and reused tail storage: about **104.063 KiB**, excluding any additional metadata or padding introduced by the redesign. This is a storage calculation, not a proven pipeline or a measured two-block kernel.

At 384 threads per CTA, registers can still prevent two CTAs from residing together. Verify compiled register allocation, spills, per-block limits and actual occupancy. Role-specific register redistribution is a secondary tool, not a substitute for reducing total demand. `setmaxnreg` moves registers within a CTA pool; lowering the producer allocation does not automatically return registers to other CTAs [R12].

**Smallest experiment:** Compare the current deep one-CTA pipeline with one resource-feasible compact design, not a broad parameter sweep. Keep both KV heads in each CTA to preserve the 128-byte-row gather advantage.

**GATE:** No official check until a static budget and local compile show a plausible residency change without spills. Stop if the second CTA is blocked by registers or if reduced staging stalls outweigh the extra independent work. More occupancy is a hypothesis, not the objective by itself.

## 7. A3 — In-kernel split merge for sliding verify

**Sources:** associative merge [R7]; existing q_len=1 cluster implementation and current run-log attribution.  
**Hypothesis:** Remove the separate global-partial/merge/norm-gate tail for short sliding-window verify workloads.

The published merge algebra is already present in the project. The new work is adapting the transport and finalization to **eight verify tokens**, not rediscovering online softmax.

Use a small 2- or 4-CTA cluster. Push each row's `(shift, denominator, output accumulator)` into its owner CTA, synchronize with valid cluster lifetime/visibility ordering, merge locally, and apply the norm-gate. Reuse dead staging memory.

**Important scaling:** 128 query rows × 66 FP32 values require **33,792 bytes per split**. Simply appending that to the current 213,104-byte allocation exceeds the documented per-CTA limit even before other overhead. Receive-buffer capacity also depends on how many rows each owner receives across the cluster. Derive that capacity; do not allocate a full all-splits tensor in every CTA.

The log's 16–24% merge fraction is a bound on the opportunity, not a predicted speedup: cluster communication and residency losses replace some of the removed work. Do not infer savings from a noisy profile sum.

**Smallest experiment:** Two cluster configurations on sliding verify b1/b16/b32, with the full-context and unsupported-head-count routes unchanged. Check empty splits, causal masking within the eight-token block and new cache writes.

**GATE:** Require a clear whole-path saving—roughly 2 us or a high-single-digit percentage on targeted sliding cases is a useful investigation threshold. Stop if cluster occupancy or communication consumes the gain. Recompute the six-load score before integrating.

## 8. M1 — Hot/cold expert tile selection without changing routes

**Source:** tile-utilization analysis in [R8]; contrast with the load-balance mechanism of [R4].  
**Status:** Already queued. This is a more precise experimental specification.

**Hypothesis:** Keep 8/16-token tiles for low-frequency experts, while processing more tokens per loaded/dequantized weight for genuinely hot experts. This targets weight reuse rather than a persistent-block start-up cost already found to be small.

Try one 32-token candidate first and one 64-token candidate only if the register budget is credible. Select work classes from device-side route counts. A fixed set of launches may consume device-generated lists; do not synchronize counts to the host or specialize on fresh input values in Python. Padding lanes must be masked, and all actual routes must be retained.

Tune the token tile jointly with K splitting. Do not assume larger tiles imply no split-K: the measured no-split path was substantially slower. Preserve all three BF16 rounding boundaries.

Measure actual L2/DRAM traffic if available; otherwise use controlled repeated-expert versus dispersed-expert routings. Unique weight bytes divided by time cannot identify whether repeated requests were served from L2 or HBM. Include one dominant expert, several hot experts, diffuse routing, and partial-tile expert counts across multiple seeds.

**GATE:** Keep only if the full MoE graph improves in hot-expert cases without meaningful regressions on the other five loads. A 10% latency reduction confined to the single `bs=256` load raises the six-load MoE score by only about **1.77%**. This is why a dramatic hot-expert GEMM result may still be near score noise.

**Do not copy SonicMoE token rounding.** It can change actual token/expert assignments; it is not just zero-padding the same math. Our proposal changes execution tiles while preserving the specified routes.

## 9. M2 — TMA for packed W4 weights, with a net resource budget

**Sources:** [R11–R13]; pipeline context [R8].  
**Status:** Already queued. Do not introduce a simultaneous GEMM-algorithm rewrite.

**Hypothesis:** Loading larger contiguous packed-weight tiles asynchronously through shared memory can reduce register/address-issue pressure sufficiently to improve the current kernel.

The current path loads weight fragments into registers. TMA transfers into shared memory, not directly into the consuming registers. A replacement introduces a shared-memory buffer, shared-to-register reads and completion synchronization. It does not natively dequantize NVFP4 for H200 or make the cost disappear.

Preserve the lossless packed representation, scale interpretation, MMA math, K split and BF16 materialization boundaries. Compare the current transport with only a small number of TMA stage sizes. Include weights and their scale traffic in the model.

**GATE:** Require either measured full-GEMM improvement or a convincing reduction in resource pressure that persists in the full MoE graph. Stop if additional shared-memory traffic/residency losses absorb the benefit. TMA-only bandwidth is not sufficient evidence.

The batch-8 path is already near its measured streaming comparator. This is a bounded transport experiment, not grounds to reopen all earlier layout/persistence searches.

## 10. A4 — Row-granular attention TMA: promising mechanism, unfavorable granularity risk

**Sources:** [R3] Section 3.2.1, [R11–R13].  
**Hypothesis:** A specialized TMA row loader might reduce part of the remaining loader-side address/copy overhead while retaining the 128-byte combined-KV-head access.

FlashInfer explicitly uses conventional async copies for scattered/non-affine KV access. Hopper TMA does not make an arbitrary page table a dense tensor tile. In addition, PTX `tile::gather4` requires Blackwell-class targets, not H200 [R12].

The feasible H200 proposal is **one contiguous physical row per command**, after reading its page-table entry. Keep K and V separate and preserve current precision. Verify that the destination layout can be consumed by WGMMA for both heads without an expensive reshuffle. Do not assume that choosing a 128-byte swizzle automatically reproduces the current head-separated/core-matrix layout.

For a 64-token tile, K and V together contain 16 KiB. There are 1,024 lane-level 16-byte copies in the current approach, versus 128 independent 128-byte row commands in the TMA sketch. These are **not comparable instruction counts**: warp-issued copies operate across lanes. TMA command throughput and tiny-copy overhead could make the change slower despite fewer scalar copy descriptions.

**Smallest experiment:** Compare (a) gather only, (b) gather plus existing V conversion/layout, and (c) complete FP32-mode attention. Keep prefetch distance and input page tables fixed initially. Use an address-layout CPU emulator before spending a check.

**GATE:** Stop if combined loading/conversion does not improve, or if only the synthetic gather wins. Prefer M2's larger contiguous transfers as the simpler initial TMA test. This candidate remains genuinely untested, not automatically good because TMA helped dense attention in a paper.

## 11. A5 — Inter-warpgroup staggering rather than another paired-tile loop

**Source:** [R1] Section 3.1, distinct from its intra-warpgroup algorithm in Section 3.2.  
**Hypothesis:** In the FP32 version, offset the two KV-head consumer warpgroups so one performs softmax while the other issues tensor-core work, reducing simultaneous demand on the same execution resource.

Do not serialize entire heads. Add only the minimum ordering necessary to stagger phases, retain independent progress where safe, and inspect SASS to confirm overlap. Keep asynchronous groups retired before problematic loop back-edges unless a separately verified implementation avoids the recorded C7514 serialization.

This is different from the paired-tile variant that tied in the bench but lost on sliding layers. Still, it has no value if the loader is already the critical path or the scheduler already achieves equivalent overlap without added barriers.

**GATE:** Admit only after FP32 phase timing shows consumer-side contention. Stop after one controlled comparison if extra barriers do not improve whole-path latency.

## 12. D1 — Length-aware GPU work partitioning: evidence required

**Sources:** [R3] planning algorithm, [R7] uneven attention partitioning, [R4] hybrid scheduling.  
**Hypothesis:** Unequal request lengths leave a measurable underfilled execution tail even after existing split tuning.

Retain full 128-byte combined-head gathers. Allocate work according to actual context/window lengths, preserving every key and a correct partial-softmax merge. Planning derived from fresh lengths must be performed inside the timed GPU computation unless the contest explicitly supplies the needed metadata. FlashInfer's CPU plan amortized across layers is not free work available to this single-kernel contract.

First measure task-duration spread and the tail, or build a conservative work-count model from metadata in a diagnostic kernel. A global 8-to-32-split sweep that was flat does not prove ideal scheduling, but it makes a blind new scheduler difficult to justify.

**GATE:** No GPU implementation check until there is a significant tail to remove. Stream-K targets balanced iteration work; it does not inherently reduce repeated hot-expert weight traffic and may hurt cache reuse. Keep any future Stream-K experiment limited to a demonstrated tail, with communication overhead counted.

## 13. B1 — Hardware-compressible BF16 allocations: an inexpensive probe if supported

**Source:** NVIDIA Hopper inline compression and CUDA compressible-memory allocation documentation [R11, R14]. This is an architectural lead, not a performance result from an attention paper.

**Hypothesis:** Transparently compressible memory backing could reduce BF16 weight traffic without the software decompression, escape-descriptor and shared-memory costs that made the earlier compression kernel unattractive.

This is not nvCOMP decompression and not another software 12-bit decoder. The tensor still contains the same BF16 bits. Support and useful compressibility of these weights are unknown.

In a preparation-stage diagnostic, query `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`. If supported and allowed by the contest's API constraints, create a separate allocation requesting `CU_MEM_ALLOCATION_COMP_GENERIC`, and verify the actual allocation properties with `cuMemGetAllocationPropertiesFromHandle`. Keep a normal-allocation control and a safe fallback. Never alter the harness allocator or supplied tensors in place.

Copy legal weight data losslessly in `prepare`, validate bit equality and lifetime/graph safety, then compare identical kernels reading normal versus compressible backing. Begin with the large BF16 pooler, not already tightly packed NVFP4. Include a control to distinguish allocator effects from data-dependent compression. Allocation support does not establish that random-looking BF16 weights compress usefully.

**GATE:** Unsupported capability or unavailable allocation attributes ends the probe. If a paired cold-L2 benchmark shows no repeatable whole-projection benefit, close it after that comparison. No expected percentage is assigned.

## 14. Numerical/architectural reserves — not ready to schedule

### N1 — Partial FP32 polynomial exp2

FA4 Section 3.1.3 distributes some exponential work to ordinary FMA instructions [R2]. Its degree-3 error argument relies partly on subsequent BF16 rounding; that is not automatically appropriate for this FP32-softmax/FP16-P.V path.

Only consider a higher-accuracy polynomial after profiling establishes MUFU as the remaining critical resource and the precision interpretation is settled. Test a small fraction of entries rather than replacing all exponentials. CPU-test range reduction, negative values, subnormals, overflow, negative infinity, masked zeros and long-context coherent-value accumulation. Approximate math remains approximate even if every instruction uses FP32. No claims of legality or accuracy parity without the relevant checks.

### N2 — Native FP8 P.V through WGMMA

The previous FP8 P.V failure used a different instruction path: `mma.sync` lowered to unpacking and half-precision operations. Native FP8 WGMMA changes that mechanism, so a revisit would not be identical. However, it would require a new probability representation and suitable V layout/transposition; both performance and precision compliance remain open. Keep this behind an explicit rules interpretation, not merely a passing relative-error test. [R1, R12]

### Optional M3 — Hot-weight multicast

If M2 wins and M1 still shows substantial redundant L2 traffic, consider a small CTA cluster that shares packed weight loads across token tiles. Multicast saves transport, not automatically repeated dequantization or all shared-memory traffic. It adds cluster residency and ownership constraints. Do not begin before the simpler prerequisites win. [R11–R13]

## 15. What not to put into the immediate backlog

**Fixed-shift softmax without recovery:** FlashDecoding++ includes detection of values outside its assumed range and recomputation using a stable method [R5, Figure 6]. It should not be dismissed as necessarily incorrect, but omitting that recovery is not the paper's algorithm. A correct version must count validation/recomputation and tolerate unfamiliar inputs. Prefer A1 first.

**Changing actual MoE routes to fill tiles:** SonicMoE's token rounding is a router/model co-design, not a padding-only kernel optimization [R8]. Preserve real routes here.

**Moving routing weights before the per-expert RMSNorm:** This is a tempting general MoE linearity trick, not a valid Muse transformation. The required form is `g * RMSNorm(y * (1+w))`. In general `RMSNorm(g*y*(1+w)) != g*RMSNorm(y*(1+w))`; normalization removes much of a positive scale. Fixed BF16 boundaries add another constraint. Do not treat generic MoE fusion diagrams as an exact specification of this model.

**A full ThunderKittens rewrite:** Its abstraction and resource tradeoffs are useful references [R6]; replacing a working codebase does not itself establish a bottleneck reduction.

**FLUTE-style dequantization work without a new resource argument:** Offline reordering and lookup-oriented kernels [R9] are interesting, but the existing kernel already repacks fragments and reports a small dequantization-only cost. Do not change NVFP4 to another quantization scheme.

**Blackwell-only instructions:** TMEM, `tcgen05` and the newer gather4 mode are not interchangeable with H200 WGMMA/TMA. Borrow algorithms, not unsupported instructions. [R2, R12]

**POD-Attention inside this kernel track:** It combines independent prefill and decode work [R10]. The current benchmark does not provide that mixed workload. Attention and MoE for the same layer/token are data-dependent, not an equivalent substitute. Reserve for serving-track scheduling.

**Repeating failed persistence/layout searches:** Require a new resource model or bottleneck observation. New paper terminology is not new experimental evidence.

## 16. Score and measurement discipline

For load `l` with shapes `i`, use:

```text
T_l = sum_i(weight_i * candidate_latency_i)
B_l = sum_i(weight_i * baseline_latency_i)
score = exp(mean_l(log(B_l / T_l)))
```

For an estimated improvement `delta_i`, calculate the score by replacing `candidate_latency_i` with `candidate_latency_i - delta_i` and recomputing the entire formula. Do not average shape speedups or add profiler durations across overlapping PDL kernels.

A 10% latency reduction on one of six equally weighted loads yields `(1/0.9)^(1/6) = 1.017715...`, approximately a 1.77% score increase. If it affects only one shape inside that load, its effect is smaller. Paper peak TFLOPS and raw gather bandwidth are not scoring formulas.

Every experimental record should contain:

- Candidate ID, code revision, changed mechanism and exact control.
- Registers, spills, static/dynamic shared memory, grid/cluster layout and preparation time.
- All test results, precision boundaries, error against the appropriate reference, and fallback activation.
- Whole-graph latency by shape, the recomputed six-load score and official confidence interval.
- Multiple routing/page-layout seeds where feasible, including distributions unlike the tuner inputs.
- A keep/close decision tied to the original hypothesis.

Use cold L2 and paired comparisons on the same GPU. Do not infer a real improvement from overlapping cross-run confidence intervals alone, nor require non-overlapping CIs as the only possible statistical test. A same-run paired diagnostic plus repeated final checks is stronger than a small difference between unrelated routing seeds.

Compiler/SASS checks, layout emulation, byte-budget calculations and the A1 algebra test can be performed without a local NVIDIA GPU. Include only a small number of justified candidates in any one remote benchmark; protect the preparation and total-run limits.

## 17. Proposed next allocation of attention

1. Finish G0; record whether loaders, consumers or storage residency now determine time. Preserve the best legally defensible fallback.
2. Do A2's static lifetime/register audit immediately because it consumes no official GPU check. Inspect feasibility before implementing.
3. If FP32 consumer work dominates, take A1 first; if loaders dominate, evaluate the most credible A2 design. A3 remains an independent short-window experiment.
4. Let the queued MoE agent execute M1/M2 with the refined controls; no generic Marlin-replacement restart.
5. Piggyback B1 capability diagnostics only where cheap and safe. Defer A4, A5 or D1 until their admission evidence exists.
6. Port an accepted standalone win into the integrated layer only after preserving all precision and graph-capture contracts. Measure the layer rather than assuming full transfer.

**The backlog is a menu with admission gates, not a request to launch every experiment.**

## References

[R1] Shah et al., **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**, arXiv:2407.08608v2. Relevant: Sections 3.1–3.3, Algorithms 1–2, Appendix B. https://arxiv.org/html/2407.08608v2

[R2] Zadouri et al., **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling**, arXiv:2603.05451v1. Relevant: Sections 3.1.3–3.1.4. Blackwell-targeted evaluation. https://arxiv.org/html/2603.05451v1

[R3] Ye et al., **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving**, arXiv:2501.01005v2. Relevant: Sections 3.2–3.3, planning algorithm. https://arxiv.org/html/2501.01005v2

[R4] Osama et al., **Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU**, arXiv:2301.03598v1. Relevant: Algorithms 4–5, Sections 5.2 and 6. https://arxiv.org/html/2301.03598v1

[R5] Hong et al., **FlashDecoding++: Faster Large Language Model Inference on GPUs**, arXiv:2311.01282. Relevant: Section 3, Figure 6 and recomputation description, Section 4. https://arxiv.org/pdf/2311.01282

[R6] Spector et al., **ThunderKittens: Simple, Fast, and Adorable AI Kernels**, arXiv:2410.20399v1. Relevant: Section 3.2, Appendix B.1. https://arxiv.org/html/2410.20399v1

[R7] Sanovar et al., **Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers**, arXiv:2405.10480. Relevant: Section IV, Figure 4. https://arxiv.org/pdf/2405.10480

[R8] **SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations**, arXiv:2512.14080, retrieved version available October 27, 2026. Relevant: Algorithms 1–3, Sections 4–5 and Appendix B. https://arxiv.org/pdf/2512.14080

[R9] Guo et al., **Fast Matrix Multiplications for Lookup Table-Quantized LLMs** (FLUTE), Findings of EMNLP 2024. Relevant: Section 3. https://aclanthology.org/2024.findings-emnlp.724.pdf

[R10] **POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference**, ASPLOS 2025. Relevant: SM-aware scheduling. https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf

[R11] NVIDIA, **Hopper Tuning Guide**. Relevant: Sections 1.4.1.1–1.4.1.3 and 1.4.2.3. https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html

[R12] NVIDIA, **PTX ISA**. Relevant: `wgmma.mma_async` numerical behavior and ordering; `setmaxnreg`; `cp.async.bulk.tensor` target requirements. Documentation evolves; use the contest compiler's supported subset. https://docs.nvidia.com/cuda/parallel-thread-execution/

[R13] Colfax Research, **CUTLASS Tutorial: Mastering the NVIDIA Tensor Memory Accelerator (TMA)**. Relevant: descriptor construction, transactions/barriers, layouts and multicast. https://research.colfax-intl.com/tutorial-hopper-tma/

[R14] NVIDIA, **CUDA Programming Guide — Virtual Memory Management**, archived CUDA 13.1.0, Section 4.16.5.2, Compressible Memory. https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/virtual-memory-management.html

## Triage (2026-10-28, against the repo)

- **G0 fence audit: acted on.** `attn_ws.cu:456` drops the loader-side `fence.proxy.async` in
  variants with bit 2 (var 2, 18, ...), relying on the math warps' fence after the barrier
  acquire. The documented CUTLASS pattern fences on the writing side before the arrive; passing
  tests do not prove the reordering is safe. Any attention submission uses the fenced variants
  (var 0 for f16, var 16 for FP32), at a measured cost of about 3% on full-context verify
  b16/b32 (bench a8a4666a91d54d68: var 0 344.6/186.8 us vs var 2 333.8/178.8 us).
- **G0 terminology:** noted. Describe the variant as FP32 accumulator mode with FP32 softmax.
- **A1 bounded-shift softmax: new.** Algebra check passes here too
  (`tools/attn_bounded_shift_check.py`, max relative L2 5.8e-15). The current kernel skips the
  rescale only when the max is unchanged (`attn_ws.cu:758`). Admit after the FP32 bench shows
  whether consumers or loaders set the pace.
- **A2 smem lifetime compaction: new.** Static budget audit costs no check; do it next.
- **A3 in-kernel sliding merge: already listed** by the attention agent as a next step.
- **M1 / M2: running** (MoE agent, TMA loads and hot-expert tiles).
- **A4 row TMA: already listed** by the attention agent; the gather4 caution (Blackwell-only)
  is new.
- **A5, D1, N1, N2, M3: gated** as the backlog says.
- **B1 compressible allocations: new probe.** Cheap capability query; expect little on Gaussian
  BF16 weights, unmeasured.

### A2 audit result (2026-10-28): closed, no check spent

Compiled with `tools/attn_ws_quick.sh` (CUDA 12.8, sm_90a) at commit 025aaec of the FP32 branch:
var 0 (f16, fenced) 106 registers, var 16 (FP32, fenced) 115, var 48 (FP32, overlapped) 128; no
spills. Blocks are 384 threads (256 math + 4 loader warps). Rounded to the 8-register allocation
unit, one block holds 43,008 / 46,080 / 49,152 registers; two need 86,016-98,304 against 65,536
per SM. Two blocks require at most 80 registers per thread, about a third below the FP32
variant, which holds 32 FP32 scores, 32 FP32 output accumulators and 16 P registers per math
thread before addresses and loop state. Escape routes checked:

- setmaxnreg: launch at 80 (pool 30,720 per block), loaders down to 24 (3,072), leaves 104 per
  math thread; the variants need 112 (f16) and 120 (FP32) after rounding. Does not fit.
- Two loader warps (320 threads): limit 96 per thread, still under 112, and loaders already
  set the pace.
- One KV head per block: gives up the 128-byte combined-row gather, the source of the gain.

Shared memory could compact to about 104 KB, but registers block a second block first. Within
one block the freed memory buys nothing measured: 8 stages ran 333.2 us vs 7 stages 333.8 us at
b32 full (bench a8a4666a91d54d68).
