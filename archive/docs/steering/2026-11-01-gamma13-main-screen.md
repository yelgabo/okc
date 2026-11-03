# Fixed MAIN gamma13 screen: secondary resource failure, stop

November1 local / November2UTC2026. Follow-up to `2026-11-01-legal-frontier-research.md`, first report published/remote-verified at553e36fcd10866b954a6171995a61ddff319f69a. Captain retains README/index ownership. No GPU/driver APIs, host linking, production changes, quota or worker fixture work.

## Decision

**Stop the frozen gamma screen.** Actual NVCC13.0.88 is present but cannot preprocess without a host compiler. The explicitly limited fallback NVRTC13.0.88 + PTXAS13.0.88 passes baseline/OFF equivalence and fails ON resources in both bool entries. This does NOT establish target NVCC equivalence or invalidate/replace the old12.8 closure. It supplies no reason to port gamma, renew short masks, install toolchains, or spend worker/GPU time on this frozen treatment.

## Fixed scope and actual result

Compile run02:29:31.802528 through02:30:23.830240UTC November2, about52seconds; entire screen/preparation/report effort remained below20minutes. Exactly MAIN only, three arms, each both `okc::moe_fused<false>` and `<true>`. No cap/mask/spelling search. Stopped at first ON resource rejection, no follow-up compilation.

|Per-entry resource|Baseline ba9|OFF e179|ON d675|
|---|---:|---:|---:|
|Registers|96|96|96|
|Barriers|16|16|16|
|Stack bytes|8|8|**16**|
|PTXAS spill-store bytes|20|20|**60**|
|PTXAS spill-load bytes|52|52|**88**|
|maxntid / minnctapersm|576 /1|576 /1|576 /1|

Both bool entries have identical resource rows. OFF normalized PTX, full normalized SASS opcode/operand sequences, resources and launch bounds all equal baseline. ON no-worse gate=false: stack and both spill fields fail. These are assembler reports, not measured local-memory transactions, dynamic exposed cost, correctness or timing. c193 CUDA localSizeBytes0 is a different quantity.12.8baseline stores4/loads32 and ON44/68 are also different compiler outputs, not contradicted by this13secondary screen.

## Source identity, flags and adaptation

|Frozen source|SHA256|
|---|---|
|baseline869076B|ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab|
|OFF870703B|e17901d562b35f127c49110d222205fab254d078ae37e941a687774973a44c18|
|ON870702B|d6756bb50a0362af9c9cd03d44a88c80fa593628254ffb2a931a643fccafdcad|

Used `.build/g-minimal-rootreview/gamma/{off,on}.py`, exact bytes matching original frozen sources, and tracked `muse_decode_layer/cand/g1b4_stage_release.py`. Extracted MAIN `MOE_CUDA_SRC` via AST pure binding evaluator, not executing Python submission code. Common defines NS3,NSF3,FNST4,PRMTSC1,DQ7=1; MAIN NW16,FAST1=0,EPI4=0,WGS1,SC6W1. ON adds only `-DOKC_FORK_GAMMA_EFFECTIVE=1`. No launch-bound changes or fast-math flag.

Existing CPU-only NVRTC driver reused in a separate helper, not edited in tools. Replaced only hardcoded define assembly and cache/include/compiler paths for13. NVRTC options compute_90a,C++17,line-info,device-as-default-execution-space; PTXAS sm_90a,-O3,-v,--warn-on-spills. The real native NVCC recipe uses-O3/gencode sm90a/lineinfo; these different frontends are NOT declared equivalent. Device prefix before torch/extension.h only; prior disclosed virtual stdint/type_traits headers and declaration-only wgs_cta_of forward declaration retained. No Torch host wrappers or host compiler behavior tested. Source-prefix hashes, exact flags, virtual headers and declaration are in per-arm manifests. PTX normalization removes debug/file/loc/comments/whitespace, not instructions; SASS normalization retains register operands/predicates/targets, not binary scheduling/encoding equality.

NVCC probe used --ptx only, no host link. Actual error: `gcc: No such file or directory; nvcc fatal: Failed to preprocess host compiler properties.` Inventory found no gcc/g++/clang/clang++/c++. No installation or substitute compiler attempted. Thus exact target compilation remains unresolved; fallback failure is scoped to the documented device-only NVRTC adaptation.

## Toolchain and preserved evidence

Cache `/root/.cache/okc-cuda130-88`. NVCC and PTXAS report V13.0.88, build cuda_13.0.r13.0/compiler.36424714_0. Loaded libnvrtc.so.13 reports API13.0; installed nvidia-cuda-nvrtc package metadata13.0.88. Runtime/include13.0.88, CCCL and nvdisasm archive13.0.85. Full executable/library paths, SHA256 and version outputs are in `evidence/2026-11-01-gamma13-main-screen/summary.json`, including nvcc/cicc/ptxas/NVRTC/builtins/nvdisasm. NVCC SHA b8e2347d9d7fcadf8812a27324d515de99d033d3e48611811ece7e66d7895f8a; NVRTC a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11; PTXAS daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2.

`compiler-artifacts.tar.gz` preserves complete three CUDA inputs, raw/normalized PTX, CUBIN, SASS, compiler logs/manifests, helper, adapted driver, full comparison and nvcc failure. SHA256932c065afef9be9db13193386277416f8de1ba2073a76049ed3b387ade6e5546. No CUBIN executed. Raw artifacts remain under `.build/research-gamma13/artifacts`; manifests retain original pre-archive output paths, which is a provenance path change, not another build. Archive compactly preserves roughly120MB uncompressed evidence instead of expanding generated SASS in Git.

## Broader decision refinement

Ownership remains a source/design question, not an implemented candidate. The first report correctly excluded T1/T8 helper routes but did not quantify the narrower selected load subset. Added reproducible `scope_sizing.py` and `scope-sizing.json` using c193 owning JSON, exact adjusted medians and rounded profile expert durations. T32B4/T128,T256MAIN correspond to decode32/verify16/verify32, both contexts. Hypothetically shrinking ALL expert duration in only these cells by10%,15%,20%,25% models whole-six1.02616,1.04005,1.05452,1.06963. These are optimistic scope scenarios, NOT bounds or epilogue forecasts: offload changes only part of expert time and adds mailbox/producer cost. Reaching the cutoff would need over20% modeled expert reduction in this subset, much more than merely finding a few scalar instructions.

Next decision: no gamma port/GPU task. Pursue ownership only if a concrete legal mailbox/progress design and sufficiently large exposed nonlinear-time evidence overcome the documented headroom/traffic obstacle; otherwise seek a broader mainloop/weight-consumption mechanism rather than declaring this an adequate roadmap. Worker owns ongoing pair#3 and all dispatch; no new prerequisite or authorization follows. Captain indexes these reports. Scoped negative recorded here (researcher may write only own reports/evidence, not shared ruled-out/index).
