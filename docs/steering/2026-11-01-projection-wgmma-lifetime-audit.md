# Owned C2 projection WGMMA lifetime audit

2026-11-01. Main evidence checkpoint `af3b550`; reviewer base `1ad42af`.
Read current coordinator,0630 outcome, projection runs, weighted-dispatch audit
and ruled-out gates. No GPU execution/check/submission, changed kernel body,
C3 repeat, stage/tile/tuner sweep or external implementation imported.

## Decision

**Scoped closure: no MoE-style misplaced steady-state wait0 was found in owned
C2 projection.** Exact source and actual local sm90a SASS already pipeline two
groups transiently, wait until at most one remains, then release the previous
operand stage. The only wait0 drains at the terminal epilogue. Short paths use
mma.sync or CUDA-core instructions, so their waits are not WGMMA group drains.
No distinct high-value kernel hypothesis or quota request follows from this audit.
This does not close all projection optimization or establish opaque cuBLASLt internals.

## Pinned owners, SGLang and active scope

Owned public C2 job `9eba96894ee1402e`: H200,PASS9,
**1.5278064829557674**,95%CI[1.504889385711431,1.5507235802001038].
Root `.build/today/projection_best.py` is the persistent frozen C2/source3044ee7
comment-repeat file, raw SHA256
`5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4`.
It is AST-identical to the C2 candidate; do not substitute current tracked C3.

The C2 receipt supplies all30 actual adjusted medians, paired SGLang medians
and task floors. SGLang's independently profiled hotspot map is owning job
`bde8b4c49c4a46bd`; it supplies kernel identities, not a paired component timing
experiment against this C2 bank. All30 profile maps are preserved in evidence.

Actual C2 profiles select **3 owned WGMMA cells**,7 large cuBLASLt cells,
and20 short cells:

|Cell|Actual owned path|C2 us|Paired SGLang us|Task floor us|SGLang profile hotspot|
|---|---|---:|---:|---:|---|
|QKV128|wd<1,128,3,1,false,8>, policytrue,var0|14.664393|16.896001|8.424107|nvjet splitK GEMM + splitKreduce|
|router128|wd<1,64,8,1,true,8>, policyfalse,var6190|6.638868|16.287999|2.211840|two hi/lo splitK GEMMs/reductions + combine|
|router256|wd<1,128,8,1,true,6>, policyfalse,var6190|9.261433|17.408000|2.676053|two hi/lo splitK GEMMs/reductions + combine|

All large BF16 cells except QKV128 use validated full-K FP32-compute Lt plans;
even pooler256's old CUTLASS fallback is not the active receipt path. Lt kernel
SASS was not returned by the official profiler and cannot be reconstructed from
a kernel name. Local crosscompilation of our source says nothing about those
vendor entries' exact outstanding-group schedule. The same limitation applies
to trying to infer runtime stalls/utilization from static instruction counts.

## Source and compiled lifetime proof

Exact C2 Python anchors: `_WD_SRC`2480; `wg_fence`2497; `wg_commit`2501;
`wg_wait`2505; producer loop2850; consumer body2875-2935; terminal partials2938;
final cluster exit3195. `_build_ext`4259 requests-O3/sm90a, no fast-math.

1. The producer waits for a stage's empty barrier before issuing new TMA writes.
   The full barrier tracks both operand tiles. Consumer warps acquire the full
   phase before issuing shared-descriptor WGMMA reads from that stage.
2. Each iteration issues four WGMMA instructions for BF16 or eight for hi+lo,
   then one commit. `wg_wait<1>` completes all older groups; this iteration's
   group may remain pending. The next iteration can issue while it remains,
   temporarily making two groups. There is no inner-loop wait0.
3. After wait1, each consumer warp's lane0 arrives on the **previous** stage's
   empty barrier; its expected count is4*WG*CLR. Previous WGMMA operand reads
   are complete before that stage can be reused. The current stage stays live.
   Producer/consumer phase and ring indices remain unchanged.
4. Final wait0 completes all outstanding operands/accumulators before the
   partials overwrite dead ring storage and before FP32 reduction/final BF16
   conversion. The final cluster lifetime barrier remains. This drain cannot
   be removed just because a correctness test passes.

The archived [PTX ISA8.7 wgmma.wait_group definition](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
specifies completion of all older groups beyond the retained count; wait0
completes all committed groups. This is the relevant completion/lifetime rule,
not an analogy to MoE's register-A/pending-publication pipeline.

Actual CUDA12.8 NVRTC/PTXAS sm90a compilation of unchanged selected device bodies:

|Selected entry|HGMMA static instructions in loop|wait1 / terminal wait0|Registers|Stack/spills|
|---|---:|---:|---:|---:|
|QKV128 wd|4|1 /1|90|0 /0|
|router128 wd|8|1 /1|58|0 /0|
|router256 wd|8|1 /1|90|0 /0|

The wait0 is outside each backward loop branch. Example QKV128: HGMMA0x0d80,
wait1 0x0d90, previous-stage arrival0x0da0, backward branch0x0dc0,
terminal wait0 0x0e20. Router128 uses wait1 0x11d0/backedge0x1200/wait0 0x1230;
router256 uses0x12c0/0x12f0/0x1320. Full snippets and SASS are retained.
No C751x serialization warning; absence of warnings alone was not the proof.

The same compile emits8 distinct selected SK templates,2 RD and9 RT combinations
(only3 RT combinations are active). Every one has HMMA, no HGMMA/group waits,
zero stack/spills. Selected shared SK geometries deduplicate10 operator names
into8 templates. Registers vary by template; no low-register/runtime-occupancy
gain is claimed. CUDA-core GEMV is source-audited but not instantiated in this
compile. Source includes no WGMMA in SK/RT/RD/GEMV bodies.

## Reproducibility and compiler limits

Tools: `proj_wait_export.py`, `proj_wait_compile.py`, `proj_wait_analyze.py`.
The exporter pins C2 raw SHA, imports it without calling prepare/kernel, chooses
existing active SK/WD instantiations and uses existing SK_DEVICE_ONLY guards.
It excludes Torch host wrappers and unrelated GEMV/CUTLASS/Lt instantiations;
no device algorithm, fence or operand instruction is stripped. RT's9 existing
combinations are explicitly labeled, not mistaken for9 scored cells.

Compilation uses cached official NVRTC12.8.93/PTXAS12.8.93, compute_90a/sm_90a,
C++17/-O3. Minimal virtual `stdint.h` fixed-width declarations and `stdlib.h`
size_t/ptrdiff_t declarations adapt the host-less header environment. First
attempt failed missingstdlib.h, then passed with that disclosed declaration-only
adapter. No GPU, driver admission or real Torch host extension compile is claimed.
Both headers, actual flags and generated/compiled hashes are in the manifest.
The owning C2 log reports **Torch CUDA13.0 and cuBLASLt130101**. Its actual
extension CUBIN/PTXAS version was not retained. Therefore this local12.8 SASS
is a source-preserving compiler gate, **not the official H200 CUBIN**; source
shows the intended wait1 lifetime, while a toolchain-specific inserted drain
on the official build remains unmeasured. Do not erase that evidence gap.

Evidence: `docs/steering/evidence/2026-11-01-projection-waits/` contains readable
device CU, export/compile manifests, logs, all22 entry resources,30-cell audit
and compressed PTX/CUBIN/SASS **evidence**, never a submitted opaque payload.
Generated CU SHA `f86b8064b33c1f516144392e4633bf54a2bdffc6985662f022cf86cd23292b66`;
CUBIN SHA `10c2224aadabcc66f16cbfbeb3a1a2a1a2ba714b11afa8a62ac29ec93c8916bf`.

## Whole30-cell score and stopping gate

All cells carry weight62. Recomputed score sums five operator times per load,
then takes the geometric mean of six SGLang/load ratios: exact1.5278064829557674.
Load sums M1/8/16/32/128/256 are35.182617/37.668069/46.914076/51.720988/
67.095785/82.416369us. The following are **arithmetic sensitivities**, not measured
speedups, attributable wait time, projected improvements or ranking forecasts:

- Cutting each of the3 entire owned-WGMMA operator times10% gives factor1.007287507.
- Moving those3 entire operators to task floors gives1.043663801.
- Impossible removal of all3 operators gives1.087122334; it is included only to
  make the scope explicit, not as an achievable budget.
- Saving1us in either128 cell gives1.002505842; in router256 gives1.002036692.

Terminal drains are only a subset of those operator times. No profile/counter
measures an exposed terminal-wait budget, and current source already pipelines
the main loop. The existing deeper-group QKV128 experiment4542b2b04cb344ec had
matching90regs/0local/197760dynamic shared/39clusters but lost0.512-0.640us
versus its own depth1 controls. The push/transpose epilogue4edaa20c98904404 lost
0.352-0.480us. Raw large-tile/residency4ca25ee4eccc495c and phase ablations
e6e5bd6d0a104faf also remain closed. This audit supplies no reopening trigger.

Preserve C2; no source optimization/check proposal. Reopen only if an exact new
compiled active source exposes a steady-state drain or different operand lifetime
that can legally retain overlap, or actual calibrated component/counter evidence
shows material exposed copy/reduction/epilogue time with a distinct representation
that avoids the prior losses. Opaque Lt investigation needs actual selected binary/
plan evidence first. Do not recycle stage counts, wait-depth, tile geometry,
barrier removal or tuner expansion under a new label.

## All30 adjusted cells

The table uses C2's paired SGLang timing, not durations copied from the older
baseline profile. Full kernel reports and per-cell score factors are in audit.json.

|Operator|M|Path family|C2 us|Paired SGLang us|Task floor us|1us saved: score gain|
|---|---:|---|---:|---:|---:|---:|
|qkv_gate|1|short|9.058|14.288|7.869|0.4817%|
|o|1|short|5.036|6.752|3.499|0.4817%|
|pooler|1|short|13.516|21.248|13.986|0.4817%|
|unpooler|1|short|3.734|7.136|1.750|0.4817%|
|router|1|short|3.838|10.512|1.751|0.4817%|
|qkv_gate|8|short|9.771|14.256|7.899|0.4494%|
|o|8|short|5.476|6.624|3.526|0.4494%|
|pooler|8|short|13.419|21.280|14.022|0.4494%|
|unpooler|8|short|4.535|7.056|1.765|0.4494%|
|router|8|short|4.468|26.592|1.777|0.4494%|
|qkv_gate|16|short|9.741|13.664|7.934|0.3597%|
|o|16|short|5.366|6.720|3.557|0.3597%|
|pooler|16|short|13.800|21.024|14.063|0.3597%|
|unpooler|16|short|13.557|18.384|14.063|0.3597%|
|router|16|short|4.449|14.688|1.806|0.3597%|
|qkv_gate|32|short|10.557|13.872|8.004|0.3259%|
|o|32|short|5.869|6.688|3.618|0.3259%|
|pooler|32|short|15.084|21.952|14.145|0.3259%|
|unpooler|32|short|14.721|19.072|14.145|0.3259%|
|router|32|short|5.490|15.456|1.864|0.3259%|
|qkv_gate|128|wd|14.664|16.896|8.424|0.2506%|
|o|128|lt|6.812|7.328|3.987|0.2506%|
|pooler|128|lt|19.994|25.200|14.636|0.2506%|
|unpooler|128|lt|18.987|20.192|14.636|0.2506%|
|router|128|wd|6.639|16.288|2.212|0.2506%|
|qkv_gate|256|lt|16.941|19.216|9.771|0.2037%|
|o|256|lt|8.338|8.736|4.478|0.2037%|
|pooler|256|lt|24.434|31.136|17.371|0.2037%|
|unpooler|256|lt|23.441|24.480|17.371|0.2037%|
|router|256|wd|9.261|17.408|2.676|0.2037%|
