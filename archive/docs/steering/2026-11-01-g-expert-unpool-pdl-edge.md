# One short PDL edge: native expert to sliced unpooler

November 1, 2026. Finite G integration lane2 source audit. Exact607 source ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. No GPU/service calls, source implementation, broad harness or performance claim. Reproducible necessary bounds: `tools/attention_g_pdl_edge_bounds.py` and [JSON](evidence/2026-11-01-g-expert-unpool-pdl/bounds.json).

## Exact edge and readiness proof

The selected `_sl_tile_after_attention` calls native Tiny/Tiny2 expert, then `dense(m,unpool_slice,T,state,'un')`, then `_sl_cluster_finish`. The prospective secondary is ONLY ordinary CL1 SK2 `slice8_r192` at T1/T8, both contexts. One launch computes1024 columns; this is NOT eight sliced launches. Owning607 profiles show one matching slice8 call per short cell at4.4–4.5us.

Helper0 already executes an UNCONDITIONAL `griddepcontrol.launch_dependents` after phaseA/B and WS_JOIN (native2409), before final bc rendezvous/norm/output. MOE_HAS_PDLFalse and _pdl0 suppress host SECONDARY policy, not that device trigger. Cooperative host selection does not change this body. All compute CTAs reach the common tail, including empty/no-job CTAs; helper-warp exit does not prevent the other threads from issuing the CTA trigger. No new producer trigger is needed. A trigger is scheduling eligibility, NOT publication of m.

[PTX ISA8.7 §9.7.13.13](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol) defines all-CTA trigger-or-completion eligibility, repeated per-CTA triggers and waiting for prerequisite completion/memory visibility. [CUDA12.8 §3.2.8.6.2–3](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization) defines secondary launch/capture edges and opportunistic concurrency. This would use an ordinary programmatic SECONDARY after an unchanged cooperative PRIMARY, not combined cooperative+secondary-PDL attributes on one kernel. Primary progress never depends on secondary progress. Serial scheduling remains correct; concurrency is not guaranteed.

## Independent prefix, first read and ownership

Slice8 uses R8/NT1/NWR1/WSK8/UPW2/CL1/NSTAGE4 and128 CTAs. With smallest P1, tid0 initializes/fences barriers, prefetches tensor-map metadata, issues immutable weight/table copies, while other lanes initialize scratch. Existing every-thread dependency wait (SK2 native148) and preserved global proxy fence precede first dependent x TMA (native152). Correction readers and refills are downstream. No m value is read in the prefix. Descriptor construction uses pointer/shape, not m values. No speculative expert address is required.

A prospective implementation must pass P1 BY VALUE to this one launch and enable attribute1 only there. Existing Python bfp_SK2_PDL and C++ g_sk2_pdl process globals are not a safe scoped host-thread/state interface. Do not blanket-toggle globalPDL or add PCX pacing/stage knobs. Preserve every fence/wait/retirement and BF16 boundary. Unsupported selected backends delegate ordinary behavior and must be reported.

m stays live as the dense input; immutable weights/tables stay in prepared state. The grid-constant map is copied in launch Params. Preserve current-stream allocator and graph-pool lifetime, nonaliasing live m/output, and independent mutable workspaces. Final finish stays ordinary serialized. Actual graph capture must establish the programmatic edge; source/CPU mocks cannot certify it. Common inherited attention/coefficient repairs must be shared by later comparison arms.

## Necessary resource bound sharply limits the budget

Native Tiny/Tiny2 NS3 weight slots require3*(32768+4096)=110592 bytes. Smem contains at least1024-byte LUT plus one8-warp/32-lane/4-float reduction buffer=4096 bytes. Thus native shared memory is at least115712 bytes, ignoring many other fields/fat buffers/alignment. Slice8 exact geometry gives XBY16384/WBY12288/SST28672/NSTAGE4; its exact host formula requires at least118240 bytes plus8*pmax. Sum233952 exceeds H200's228KiB=233472 perSM. Same-SM simultaneous residency is impossible even under these conservative lower bounds. My initial192KB-secondary cap-based guess was wrong: _r192 is a cap, not allocated size.

Local selected Tiny uses147 registers*288 threads>32768, excluding two such CTAs on a65536-register SM. Cooperative native launches SMcount CTAs. The secondary can start only on an SM after a primary CTA retires; its wait still needs the last primary to finish. Legal overlap is therefore restricted to first-to-last primary retirement spread, NOT the entire29–72us expert activity. That spread and useful prefix completion have not been measured.

First weight stage across128 CTAs is1,572,864 bytes, one of four stages. Byte fraction is not elapsed-time fraction. Full unpool4.4–4.5us includes dependent input/MMA/reduction/stores that cannot all be hidden. No bandwidth/time forecast follows.

Exact607 weighted short latencies are78.549044818/115.990822900us. Holding other four loads fixed, equal hypothetical removed time per short context gives six-load factors:0.5us→1.001785893;1us→1.003584974;2us→1.007223412;3us→1.010916763. A1% factor needs2.753183990us per short context. These are sensitivities, not gains. Evidence does not establish enough retirement spread/prefix for that material threshold.

## Decision and next group

Stop before OFF implementation or quota request on this edge: a legal construction is concrete, but meaningful exposed budget is missing and source resource bounds narrow it substantially. This is not a measured loss or general PDL prohibition. Reopening requires exact same-input retirement/prefix critical intervals with controls, not aggregate subtraction or larger-P sweeps.

The next changed G group to inspect is disabled short GLE_T/B1X O-boundary integration. Its old rewrite moves BF16 rounds and remains ineligible unchanged. A distinct alternative would materialize exact BF16 O/postnorm/residual/preFFN interfaces while changing finalization placement, requiring new complete-writer/last-owner proof. The separate b1 activity3.0–3.4us is only a screen, not exposed gain. No implementation or eligible gain is claimed. Root may prefer already concrete RMW/gamma representations; reports and control files do not count as additional performance candidates. No unrelated tuning is proposed.
