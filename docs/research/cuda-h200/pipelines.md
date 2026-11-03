# Pipelines and memory ordering

Source sections: shared memory, stages, warp specialization, BF16/FP16, W4A16 and instruction choice. Read [hardware and numeric rules](hardware-and-numerics.md) first.

## Start from the selected implementation

[The scored layer source](../../../muse_decode_layer/cand/g1b4_stage_release.py) contains several historical bodies and patched helper sources. Its final `_fork_native_moe(M, state)` and `_fork_native_ma_prepare` determine the actual build. T16 selects MID; T32 selects B4 WGS; T128/T256 select MAIN WGS. The skinny, owned fallback and WGS paths have different math instructions and resource requirements. A search hit in an unused source literal is insufficient.

For B4/MAIN, packed weights move through a raw shared ring, consumers load codes/scales and decode register operands, while gathered BF16 activations use a shared-memory descriptor. WGS overlaps parity-buffered decode with outstanding WGMMA groups, then drains before operand reuse. Generic persistence, direct register decoding and fused SiLU already exist. [The exact overlap comparison](../../steering/2026-11-02-research-mainloop-overlap-decision.md) explains the selected loop and its obligations.

The playbook's TMA-to-decoded-SMEM-to-WGMMA drawing is one possible design. It is not a requirement to replace our register-A path. One-dimensional bulk copies and tensor-map transfers have different setup costs. Measure the operand that needs a new mechanism rather than replacing every load with a tensor-map operation. WGMMA also need not replace efficient warp-level MMA in the tiny path.

## Lifetimes that an optimization must preserve

Keep packed weight slots alive through all generic readers. Keep descriptor-backed activation slots alive through all outstanding WGMMA readers. Retain register operands until their asynchronous consumers finish. Publish generic stores before async-proxy readers and preserve each barrier's arrival count and phase. Preserve cross-CTA G1-to-G2 publication, split reduction ownership, scratch reset and cooperative admission.

`wgmma.fence`, `fence.proxy.async`, WGMMA completion waits and mbarrier completion are different operations. Moving one is a memory-model change. The uploaded decision tree's "remove barriers" is not an instruction to delete ordering. Require a PTX proof of an equivalent protocol before a performance test. See [PTX 8.7 async proxy semantics](https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence) and the source study above. Sanitizers supplement this proof.

## What merits another experiment

Compare stage count, K depth, tile dimensions, independent accumulator chains and producer/consumer register allocations as a coherent candidate. Hold unrelated dispatch and numeric boundaries fixed. Record actual lowered registers, local loads/stores, dynamic shared memory, resident blocks and selected variants. More stages can hide latency or consume the resources needed by the compute path. The research's 2-to-5-stage sweep is a starting range, not a workload-independent optimum.

Derive shared-memory swizzle from operand descriptors and actual access patterns. A padding probe can test a generic loader conflict, but arbitrary leading-dimension padding can invalidate TMA/WGMMA descriptors. Verify addresses, descriptor constraints and output values before measuring bank counters. A register-side decoder does not acquire a decoded-weight shared-memory bottleneck merely because a generic example has one.

The [fixed shared-A implementation](../../steering/2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md), own job `5adf172986ff43dc`, passed target correctness and was parked after a slower descriptive comparison. The [direct-register-load implementation](../../steering/2026-11-02-driver-direct-weight-terminal-and-dense-backing.md), own job `0b684a4065354fc5`, was also parked. These close those exact configurations. Reopening needs the named new mechanism or exposure evidence in [ruled-out.md](../../ruled-out.md), rather than another generic producer/stage sweep.
