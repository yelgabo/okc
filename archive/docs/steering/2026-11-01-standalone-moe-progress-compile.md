# Historical standalone MoE progress repair: local compiler gate

2026-11-01. Reviewer base `f33ffcf`. Bounded CUDA12.8 sm90a compilation only:
no GPU execution, official call, production change, new toolchain or variant sweep.

**The first repair fails the requested no-new-spill gate on main47.** This is
compiler evidence, not a numerical failure, a measured slowdown or a reason to
retain an unsafe progress contract. Root and the owner were notified before
any follow-up decision. No automatic repair tuning or official retry follows.

## Exact pair and extraction

Control is `ec565a7:muse_moe_w4a16/submission.py`, 235700 bytes, SHA256
`b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14`.
Owner's frozen native-core draft is
`.build/today-moe/standalone-progress/draft.py`, 237536 bytes, SHA256
`7c9151276e18bb8b33b251579a3984df620f0d774f6d2892037bc460f7cb9488`.
The owner generator `tools/moe_standalone_progress.py::core` reverse-checks
only two host launchers and the shared `spin_failed` helper changed. The helper
adds fatal `trap;` after recording the watchdog diagnostic. Cooperative-launch
admission and queue bounds are host changes; this device compile does not
compile or certify those wrappers. Startup qualification remains owner work.

Reused existing `tools/moe_decoder_compile.py`, SHA256
`0e44f8e6e46dad84e99b2f6a9c0cc45870aac766420c269d8de797fd5aaafdc4`.
The only driver edit removes unused main51 from name expressions, leaving
`&moe_main_kernel<47>` and `&sk_8_3_0::moe_fused_kernel<1>`. Actual AST source
generation preserves device bodies; extraction removes Torch/ATen includes,
host-extension tail and unused host launch wrappers, substitutes CUDA runtime
and fixed-width virtual stdint declarations, and selects the existing sk8/3
namespace. No algorithm, fence, arithmetic or device flag was stripped.

NVRTC12.8.93 emits compute_90a PTX with C++17 and line info; PTXAS12.8.93
uses sm_90a, O3, verbose resource reporting and spill warnings. No fast-math
flag is added. Installed pinned CUDA cache was reused. Complete options,
lowered symbols and source/device/PTX/CUBIN hashes are archived per side.
Disassembly uses existing nvdisasm12.8.90. Neither CUDA driver nor GPU runs.

## Actual selected-entry results

| Entry | Control | Repair |
|---|---|---|
| main47 registers / barriers |96 /16|96 /16|
| main47 stack / spill stores / spill loads, bytes |16 /12 /112|16 /16 /128|
| main47 static instructions |16480|16712|
| skinny1 registers / barriers |140 /16|140 /16|
| skinny1 stack / spill stores / spill loads |0 /0 /0|0 /0 /0|
| skinny1 static instructions |3000|3000|

Main47 already spills in the historical control. The repair adds4 spill-store
bytes and16 spill-load bytes reported by PTXAS; describing either side as
spill-free would be false. The main47 change is not only cold trap opcodes:
it adds73 BPT.TRAP sites and changes control flow, collectives, shuffles,
register/address allocation and local instructions. Static instruction count
is not a runtime cost measurement.

For main47, the opcode counts of all HGMMA shapes, HMUL2, FENCE, MEMBAR,
BAR, SYNCS and WARPGROUP instructions are preserved, but their full operand
sequence is not identical (register allocation/control flow changes). There
are192 HGMMA instructions,1024 BF16 HMUL2,96 WARPGROUP.ARRIVE and108
WARPGROUP.DEPBAR.LE in each compiled entry. This census is a bounded compiler
check; identical counts alone do not prove ordering equivalence.

For skinny1 the recorded math/fence/barrier instruction sequence is identical.
Its opcode delta is three BPT.TRAP additions, four fewer NOP and one added
YIELD; total instruction count and resources are unchanged. Source-level
math/fence preservation still rests on the owner's explicit reverse proof,
not this census alone.

Evidence: `evidence/2026-11-01-standalone-moe-progress-compile/` contains
the driver recipe, exact compile driver snapshot, comparison script/result,
per-side manifests and compiler logs. Full local PTX/CUBIN/SASS remain under
`/root/.cache/okc-standalone-progress/{control,repair}`. No GPU utilization,
latency, H200 residency, stream-capture or correctness claim follows.

## Decision and next gate

Stop this exact draft at the no-new-spill gate. Any differently represented
fatal path requires root coordination, a new source pin and the same bounded
two-entry compile comparison; do not sweep caps, geometry or precision.
Even a compiler pass would still require independent host/progress review,
the fixed abe765 six-load policy qualification and actual H200 gates before
a measured candidate recommendation. The separate retained C2 projection
progress report is unchanged.

## Authorized termination-contract screen: also closed

After the first result, root authorized exactly one distinct compiler-contract
screen. Private source237565 bytes, SHA256
`db0b8cc1512f85c8967924a3e0f0fc53645be482f071f74cf0257224379210af`,
adds only `__builtin_unreachable();` immediately after the existing fatal
`asm volatile("trap;" ::: "memory");`. The diagnostic atomic and memory
clobber remain. It does not edit the owner's frozen7c915 source or generator.

New justification was the documented optimizer hint in
[CUDA12.8 section7.18.5](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#builtin-unreachable),
combined with trap's terminating contract. The hypothesis was that declaring
the continuation unreachable might avoid compiler liveness costs; this was
a hypothesis, not a diagnosed cause of the first spill change.

The identical two-entry compiler gate still fails: main47 remains96 registers,
16 barriers,16-byte stack,16 spill-store and128 spill-load bytes versus the
control's12/112. Main47 has16696 static instructions and all73 trap sites;
its full recorded math/fence operand sequence still differs from the control.
Skinny remains140 registers,16 barriers,zero stack/spills,3000 instructions,
three traps and identical recorded math/fence operands. Main math/fence/barrier
opcode counts remain equal. No hot-path/performance improvement is claimed.

Archive `unreachable-comparison.json` and `unreachable/` preserve this actual
result. Close both exact termination representations; no further compiler-hint,
cap, trap-placement or arithmetic sweep is authorized by this report. The
unsafe old launch/progress contract is not thereby certified as a fallback.
