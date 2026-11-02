# P2 complete expert-engine feasibility: reject the present allocation path

November 2, 2026. Allocation **b395d49b6bd8291d888de7bbdfd5be80a4e50860**, human priority **588847c955bbcb00859ae6b289fa156e6699f7e9**. Independent source/CPU review; P1 architecture intake remains its owner's work.

## Decision

**REJECT WITH CONCRETE STORED-WEIGHT RULE CONTRADICTION / MISSING SUBSTANTIAL PATH.** P1's final report is committed at **c8aacece1b6447b577187a7d7fb5bd7d4ea7e90c**, not an intermediate draft. I agree with its finite no-allocation conclusion, with a stronger reason for Design B and narrower interpretations of its traffic/profile arguments:

- **B, prepare-time BF16 expansion, is not an admissible implementation under the available task contract.** TASK.md:102 explicitly requires NVFP4 weights to remain NVFP4, permitting lossless repacking/compression and dequantization inside kernels. Preparing a3GiB BF16 operand bank stores those weights at a different precision even if every decoded operand word is exact and globals remain post-dot. Bit-equivalence would address numerical error, not this separate stored-representation restriction. Do not allocate implementation to establish performance first or silently reinterpret the rule. P1 already rejects B on cost; this additionally blocks promotion absent authoritative changed permission.
- **A, current Humming grouped-contiguous NVFP4/BF16, is structurally distinct but lacks an instantiated, substantial complete-path advantage.** The inspected source supports the necessary formats in principle; no concrete numerical contradiction was found in P1's *stated* sequence. There is no exact selected-config/current-source Muse result or exposed-latency budget paying the extra full-operator costs. Do not turn the new library version into an automatic wrapper/harness/target allocation.

Neither outcome is a measured speed result or a rejection of the whole expert-engine space. No attention/T1 repair, follow-up dispatch, compiler or quota use is requested.

## Exact review inputs and validation

Control public **607eb52bdaf24160**, fresh unchanged **cc474883117947c0**, full layer SHA256 **ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab**, native control **c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71**, helper **b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716**. Read actual final dispatcher/preparation/epilogues, reference.py, nvfp4.py, TASK.md, P1 final/analysis/manifest, exact component-adapter closure, relevant ruled-out and old owning Humming log. TASK.md is tracked and unmodified; its last content commit is **e81d555b0667fd3b9548d72207bfb7a6833acf82**. Task/source hashes are in [independent analysis](evidence/2026-11-02-research-broad-expert-feasibility/analysis.json).

Reused P1's existing **948e6fd547e703065be51529ac1db5569256774e** Humming snapshot, without download/import/execution. Independently hash-checked and read five pinned primary files: config/mma.py, schema/humming.py, forward.py, arith/epilogue_arith.cuh and scheduler.cuh. Relevant primary locations: MMA82–122, schema325–371, epilogue63–99, scheduler grouped token-count/tile/prefix handling. The source's BF16 activation -> BF16 MMA operand choice and FP32 tensor-scale branch are compatible in principle. Schema349 preserves one FP32 tensor scale/expert, but alternate stacked-scale branches can convert/fold scales; a concrete caller/config must establish which branch is used. Supporting a dtype is not a proof of exact exponent/rounding equivalence or contest adoption eligibility.

[CPU analysis](evidence/2026-11-02-research-broad-expert-feasibility/analyze.mjs) reads source as text only:7selected-source invariants, task-rule assertion,5primary-file hashes and3independently reconstructed budget rows pass. These are arithmetic/source checks, not kernel tests or a new measuring harness.

## Actual selected semantics: MID is not MAIN

|Load|Final selection|Existing output/norm ownership|
|---|---|---|
|D16,T16|MID NW8/TINY1/SPLIT1, @moe_helper0df; alias is the unpatched helper0, DF off|At most8tokens/job; helper writes BF16 inter and FP32 gamma-weighted y; per-route sum-of-squares atomics, publication and final slot combine|
|D32,T32|B4 WGS NW16 + B4FIN, SC6 only if eligible|Gate/up epilogue writes SiLU directly to B's packed activation layout; down writes FP32 y and256norm partials/route; four1024-column final segments/token reduce the partials|
|V16/T128,V32/T256|MAIN WGS NW16, SC6 only if eligible|Same direct activation-layout store; down writes FP32 y, not per-tile squared-norm atomics. Final token-owned pass reads full y for norm, then reads it for weighted combination|

Source anchors: definitions6858–6864; final dispatch16084–16106; helper alias15248–15249; MID jobs12830–12853; WGS epilogue5803–5853; B4 final6488–6546; MAIN6548–6632; preparation10156–10184. Native admission/ordinary cooperative launch is retained; all four target T fit one existing384-token call, so no Python chunk loop is eliminated.

Both designs must preserve E2M1 codes, E4M3 block16 scales and FP32 per-expert global scales; the control's exponent-offset operand convention plus post-dot scaling cannot be replaced by indiscriminate global folding. Scale gate/up FP32 reductions then BF16-round EACH before SiLU/product, BF16-store the intermediate; BF16-round scaled down result before multiplying effective **BF16(1+raw BF16 normw)** in FP32. Keep that product FP32 through per-expert full4096-output RMS with epsilon1e-8, then FP32 slot-weighted accumulation and BF16 final expert output. The standalone old FP32-gamma interface is not this layer ABI. No proposed late norm-after-mixing shortcut exists in P1; inventing a counterexample to an unproposed shortcut would not be an independent finding.

Remote slots remain invalid; they do not become local via modulo32. Normalized top8 weights remain those from all selected experts, not renormalized over local entries. Preserve repeated-token/different-expert entries, occupied->empty replay, wholly empty expert groups/output zeros, original-slot combine order and private scratch reset/publication. Current native PRMT admission excludes block bytes outside0x08..0x7e; zero block-scale inputs belong to fallback eligibility, not evidence the native path was validated on them. Zero codes/globals still need their actual contract. No target numerical/progress claim is made.

## Removed work must be net of the control

P1 correctly retains two dense products, nonlinearities and expert-wide norm in A. Persistent routing, shared weight use, stage streaming and fused activation-layout stores already exist in the control. A's optimistic five-call path replaces one callable: routing/gather, G1, activation, G2, norm/scatter. Count/prefix/placement may require more boundaries. The claimed81,920Rbytes read/write for compact X/G1/H/Y is a **gross proposed traffic count**, not additional traffic versus control; the control also gathers/stores and reads inter/y. No claim that all that traffic is new, all absent, or automatically on HBM is justified. Conversely A materializes full G1 BF16 output that WGS currently sends through its epilogue directly into the B layout. Stage workspace, padding, split reductions, empty-tail initialization, norm output and capture lifetime remain real costs.

A more important reuse qualification: MID re-streams weights per ceil(count/8) job; WGS jobs cap at64tokens. It is wrong to transfer the T1/T8 “once/unique expert” fact to every long route. A legal hypothetical MID count16 on one expert yields TWO control weight streams; a single expanded pass would be1.778x their ordinary bytes, not3.556x. At count128 WGS has2jobs, while a grouped BM16/BM32 schedule has8/4token tiles before cache reuse. These examples establish source accounting only, not fresh route frequencies or a reopened optimization.

P1's uniform-independent routing/one-pass table is explicitly conditional, not the actual cc474 traffic. The task calls routing skewed; reference.py480–486 chooses the busiest32-expert shard, and520–537 fits router load/distinct targets. The fresh receipt does not contain per-expert counts/padded tiles. Thus the broad-cold BF16 bandwidth story is a useful warning, **not a universal impossibility proof**. It does not need to be universal: B already fails the explicit storage rule.

Old owning **ae8993f4e3e1435c**, std256/seed292 log: matched masked419.360us versus owned217.152us; the separately profiled GEMM sum380.511us/total421.599us diagnoses that exact default schedule. It is not a transferable current-Humming lower bound or an exposed saving estimate. Source948 differs from636 and configs were not retained completely; no wrapper retry or universal newer-library rejection follows.

## Independent six-load budget, with replacement costs

Rebuilt both contexts directly from the fresh owning receipt, C=(46*sliding+16*full)/62: D16 **152.016250us**, D32 **191.897812us**, V16 **261.713045us**, V32 **351.080222us**; T1/T8 unchanged. Corresponding weighted expert activities are69.561290/75.390323/146.141935/197.338710us. Activities are NOT exposed durations that can simply be subtracted.

Sizing model: gain=product(C/(C-f*E+h))^(1/6)-1 over the four affected loads, fixed baselines; h is **net incremental replacement cost**, not all gross new work. f assumes gross activity reduction fully reaches the endpoint. Requirements:

|Desired layer gain|f with h=0|f with h=5us/load|f with h=10us/load|
|---|---:|---:|---:|
|3%|8.798%|13.459%|18.099%|
|Historical sampled layer cutoff6.5763%|18.472%|23.119%|27.744%|
|10%|26.993%|31.625%|36.235%|

h=5/10us are **sensitivity assumptions, not measured launch costs**. At h=0 the3% row needs net6.120/6.633/12.857/17.361us;10% needs18.776/20.350/39.448/53.267us. If only MAIN's two verify loads improve,3% already needs15.147% of their activity in this fully-exposed model;10% needs44.388%. Do not infer MID benefit from a main-engine result or apply a1%standalone gate. The sampled cutoff is dated prior evidence, not a fresh board/overall formula or claim of top100 sufficiency.

Preparation costs are also unresolved: A packs additional banks/JIT state; B would allocate/write3GiB plus retained banks but is ineligible. Both require separate mutable metadata/output/scratch and complete replay integration. Untimed prepare is not free of300-second/memory/capture constraints. No measured launch, physical bytes, occupancy, codegen, per-expert histogram or matched current-library endpoint closes these uncertainties.

## Smallest information-producing later gate and stop

**No implementation or target gate is accepted now.** To change A's rejection, obtain one exact current-source/config same-format32expert4096-shape cold result with actual count/padding and full BF16/scale contract, sufficient to support the substantial net budget after glue; do not re-enumerate repositories. If later separately allocated, the smallest decisive timing object is ONE fixed complete expert callable including route metadata, both GEMMs, SiLU, per-expert norm/combine, scratch/reset and all boundaries, paired against exact control on matched runtime routes/private states with replay-output checks. Measure selected long loads separately; no tile sweep, profiler-sum promotion or reopening T1/attention. A component survivor would still need complete-layer confirmation. This names the discriminator; no harness is built and no check is requested.

P2 upholds P1's finite stop with the additional explicit B policy blocker, route-specific reuse correction and net-cost sensitivities. README/shared logs and P1 files remain untouched. No children, donor execution, compiler/install/GPU/Driver/challenge CLI/POST/default/public/scored action occurred.
