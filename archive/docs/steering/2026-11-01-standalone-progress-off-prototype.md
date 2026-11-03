# OFF historical standalone MoE progress prototype: blocked at compiler gate

**No qualified standalone replacement was produced.** This source/CPU prototype addresses the documented progress gap of exact historical ec565/b35, but its main47 compiler resources fail the root's no-new-spill gate. The actual public entries remain disabled. Do not dispatch, submit or flip its gates. No production file, GPU, official service or reset queue was changed.

## Exact source and limited eligibility

The generator `tools/moe_standalone_progress.py` accepts only historical235700-byte SHA `b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14`, obtained through tracked Git. It pins both reused standalone cooperative wrapper patterns. Native-core compile draft7c915127…/237536B and final **OFF prototype `68bf4c1f336611bd7f8e0356fda07b131afa35a10579ad5fd50f941e7a7be6a6` /241838B** have identical native device changes. The final adds only disabled/fixed-policy host preparation and strict checks. [Manifest](evidence/2026-11-01-standalone-progress-prototype/manifest.json).

Root's initial circular/nonzero-bank assumption was corrected by source inspection. Historical prologues use `global_id - lo` with a nonwrapping local interval, pack weights0..31 and validate that same bank. This pure prototype explicitly accepts only its proven EP8 local_start0/count32/global256/4096-width domain; unsupported metadata fails closed. It does not invent circular routing or new weight repacking. Original512-token chunk splitting remains, including513 as512+1; input contiguity/int32/float32 coercions in `_run` remain unchanged. Full256-expert local execution is not supported, as in the original native scope.

## Native repair and proof

Both actual wrappers are repaired: main `launch_main<47>` and generated `sk_8_3_0::launch<1>`. The checked existing `moe_owned_cooperative_pair.py` pattern supplies SM90/cooperative capability, actual kernel/block/shared occupancy, bounded positive grid<=SMcount and136, chunk/local bounds, fatal CUDA errors and `cudaLaunchCooperativeKernel` using the original by-value Params/stream ABI. There is no PDL field/launch option in this historical Params; the layer launcher was deliberately not copied blindly.

Main additionally checks the conservative queue capacity with int64 widening before multiplication:

```text
G * (kBqMax - 1) >= 16 * (floor(chunkM * 8 /64) + nloc), with kBqMax ==64
```

Total cooperative admission alone would not prove capacity of the per-CTA dynamic B queue. The63 usable entries reserve the final sentinel; the bound covers the worst routed job population without reading runtime IDs. All actualH200132-CTA tested chunk sizes satisfy it; smaller configurations may fail explicitly.

One shared `spin_failed` keeps its clock threshold and diagnostic atomic, then executes volatile PTX `trap;` with the original memory clobber before its old return. It therefore cannot normally return into unpublished-data consumption/cleanup on expiry. Both selected families' active full/readiness/credit/Bqueue/grid polls use this helper. No compute/packing/memory-order fence, shared/ring phase, routing table, scratch layout, native call signature or generation protocol changes. Exact reversal of the two launchers and shared watchdog recovers every original native byte.

Independent native source/progress/bounds review is `14d5687`, terminal `c00895f`, in `docs/steering/2026-11-01-standalone-progress-native-review.md`. Main phase A and skinny helper credits precede their B waits; cooperative admission removes the unscheduled-CTA cycle without creating a new dependency. Hardware WGMMA/named/CTA waits retain their convergence obligations. [CUDA12.8 grid co-residency contract](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#grid-synchronization-cg), [PTX8.7 fatal trap](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#miscellaneous-instructions-trap).

## Frozen policy and strict CPU-tested host prototype

The new prospective setup freezes the actual abe765 table: tokens1/8/16/32→skinny variant3 `(8,3,0)`,128/256→main variant9/mode47, preserving original next-bucket and chunk behavior. It does not call the old tuner, `_bench`, legacy packing or comparative timing. Old helper definitions remain in the source but the new normal entry/prepare shadow them; source fit is not a constraint here. The original unrounded FP32 effective norm parameter is retained, not silently combined with a precision change.

`moe_standalone_progress_body.py` implements an INTERNAL prospective setup using private scratch for selected self-tests. It explicitly asserts their boolean before returning any candidate state. The selected self-test now gates finite output **and reference**, global relative L2<=0.008, row<=0.075 (with zero-safe denominators), and diagnostic ctr8==0. It does not reset diagnostic counters to conceal a failure. Cases include1/4/8/16/32/33/128/257/512/513, normal/skewed. Reference caches are cleared and TF32 restored in finally. This helper is not the missing full graph qualifier.

The actual `prepare` and `kernel_fn` require three false gates: enabled, static compiler gate passed, complete fresh/empty graph qualification. All remain **False**. The normal prototype cannot run, even if someone enables only its first switch. No tool offers an enable/dispatch option.

Five real TorchCPU/source tests pass (`tools/moe_standalone_progress_test.py`): exact source pin/delta and OFF gates; actual strict checker including a row error that passes the global0.008 test but must fail0.075; finite output/reference and exact-zero checks; frozen prospective setup/no-tuner/private scratch/false self-test rejection; actual selected self-test diagnostic rejection without reset; and metadata/chunk/budget rejection. CUDA/native/model arithmetic in prospective setup are explicit stand-ins. These are not H200 self-test results or complete qualification coverage.

## Actual selected compiler result and stopping decision

Independent compiler owner reports `abd6652` and `a20ec47`, [compiler report](2026-11-01-standalone-moe-progress-compile.md), compare exact same local CUDA12.8 sm90a selected expressions. Device-only extraction removes host launchers, so it does not newly compile the cooperative CUDA host API. No GPU runs.

| Selected entry | Historical control | Trap repair |
|---|---|---|
| main47 registers /barriers |96 /16|96 /16|
| main47 stack /spill stores /spill loads bytes |16 /12 /112|16 /16 /128|
| main47 static instructions |16480|16712|
| skinny1 registers /barriers |140 /16|140 /16|
| skinny1 stack /spill stores /spill loads |0 /0 /0|0 /0 /0|
| skinny1 static instructions |3000|3000|

Main47 already spilled; the gate is **no new spills**, not absolute zero. Main gains73 static BPT.TRAP sites; math/fence/barrier opcode counts remain, but register allocation/control flow/operands differ. Skinny gains3 traps with unchanged critical math/fence sequence. Neither counts nor spills are a latency/score measurement.

Root authorized one separate optimizer-contract screen: add `__builtin_unreachable()` immediately AFTER the retained diagnostic+volatile trap. Private237565-byte source `db0b8cc1512f85c8967924a3e0f0fc53645be482f071f74cf0257224379210af` was not installed in this generator. It also reports main16/128 spill bytes and fails the same gate. Both exact representations are closed. Do not sweep termination syntax, caps, registers, helper layout or old optimization knobs after these results.

## Gated qualification design, intentionally not built

A full GPU qualifier is **not implemented or authorized** after the failed static gate. For a future distinct, root-approved mechanism that clears source and compiler gates, the minimum design is: fixed six-load policy; all5 official standalone tests/all6 benchmark cells; strict FP32 finite/global/row acceptance; actual attributes/cooperative admission; held-out normal/skew inputs with real global256 IDs; changed captured input/ID/weight buffers; empty/all-remote↔occupied reuse with exact zeros and no counter resets; scratch guards; long generation/parity replay;33/257 and512/513 chunk boundaries; private test ownership and cleanup; bounded complete setup under300s. Every expected case/replay/resource record must be present. Unsupported domain/admission/CUDA errors fail closed, never an ordinary launch fallback.

This design does not qualify the prototype or historical source. Historical abe765 **2.215711562× remains historical only**; no replacement score, speedup, utilization or improved rank is claimed. Layer job1fad70bbe3854cbc remains outside this source audit and unresolved during the coordinator's verified530 outage. First-two G reset payloads/order and production bytes are preserved. Parent owns any future delegation or official slot.
