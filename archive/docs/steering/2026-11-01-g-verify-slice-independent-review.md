# Independent private G verify32/sliding slice review

November1,2026. Read-only review of root `tools/dl_g_verify_slice.py` (initial reviewed raw hash`f39d24b9f73114ad5075ab9e844dc69a384d0331b11746d8d8f8cd70eb6b84c3`) and `.build/today/g-verify-main-slice.py`, against exact scored607eb52bdaf24160 source SHA`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. Builder is under active root edits; this report is not final-byte clearance. No source edits, GPU jobs or public activity by reviewer.

**Conditional source disposition:** the selected G computation matches the scored verify32/sliding route. The smaller semantic slice is a valid new packaging direction distinct from the closed unchanged-literal pair. Missing bindings/runtime facade and preparation/capture conformance must still pass; neither size fit nor unchanged native literals certifies a ready H200 pair.

## Concrete issues sent immediately

1. **Owned slice capacity guard is wrong for actual inputs.** Existing `dl_owned_slice.CUSTOM.kernel_fn` requires page_table width80000, while published B32/q_len8/page1/agentic ctx80000 generation allocates `ceil(80000*1.075)+8 =86008`. Its old mock used80000 and concealed this. Correct the private guard and test actual generator metadata rather than preserving a false predicate.
2. **Attention module build is not workspace warmup.** G custom prepare initially calls `_att7()` only. `_workspace7` and `_cluster_bufs7` allocate guarded per-state buffers and explicitly reject growth during capture. The paired harness must eagerly run/warm the actual selected B32/L8/sliding plan before capture, with the same state, then establish stable pointers/capacity. A module import alone does not meet that contract.
3. **Fail-closed geometry/domain guard.** G's initial custom kernel checks only batch/q_len/tokens/window/page. It bypasses the original `_supported7`, `_att_fits7` and scaled-verify fallback. Enforce hidden8192,16query/2KVheads,head_dim64,256experts,top8,local32, supported cache/input sizes and kv_scale1.0 for this exact private arm, or preserve equivalent checks. Unsupported cases may fail rather than create a private fallback, but must not silently run a different mathematical contract.

Root acknowledged the page/guard/warm issues and owns repairs. Final regenerated source identity and runtime facade are separate gates.

## Pipeline/ABI findings

The custom route is QKV256 Lt→native BF16-matched attention→O256 Lt→`_graft_row_boundary` POSTFalse/V4/NW16→FP32 concatenated pool/router256 Lt→packed1 routing→native MAIN<true>→unpool256 Lt→the same row boundary POSTTrue. This matches the actual scored selected dispatch and the four retained projection profiler rows. `_rq_gemm` is disabled in the scored source, so bypassing that null check does not choose a different GEMM.

Router FP32 weights are split into hiBF16 plus residualBF16 and concatenated after the pooler rows exactly as before. Concatenated output remains FP32; route applies the same raw `g_preexp` BF16 parameter ABI, mandatory pooler/normalization rounds, modulo local-ID mapping and FP32 normalized top-k weights. **Do not pre-add1 to g_preexp in the host state**: the route does that internally.

Graft residual coefficients use the same FP32 sigmoid/clamp/sqrt expressions. Graft pre1/post2/pre2 contain BF16(1+raw.float()), preserving the scored zero-centered-gamma boundary. Native `ma_w[7]` remains the **raw BF16 post-expert parameter**, because the native kernel forms the effective BF16 gamma internally. No gamma folding or relocated activation rounding is authorized.

`_fork_native_ma_prepare`, its packs/scratch guards, and the original main CUDA source remain selected. Main defines plusSC6W/PRMT/DQ7 match `_fork_native_moe(256,state)` through `_ma_ext`; no B4 or alternate helper source is needed. Calling the extension directly bypasses only the delegating `_RouterExt`, not its device math. Cooperative capability/grid/local-ID/dtype/shared-memory/occupancy admission remains in the unchanged C++ wrapper. Native packing/domain/resource/math failures must be fatal in this restricted arm; no fallback score may be presented as native evidence.

The original notices are retained after the semantic code selection. Literal/registration identity, the new reduced byte count, prepare time, module linking, selected native resources, pointer/private-state ownership, full routing/cache/math checks and fresh/replay behavior must be tested on the **final** composed input. This report does not certify those unexecuted gates or a full-score improvement from one sliding cell.

**Review sealed.** Root owns final guards, missing tuple/global binding fixes, runtime facade and actual capture warmup. Reviewer remains available for the separate wait-trace probe; no dispatch authorization is exercised here.
