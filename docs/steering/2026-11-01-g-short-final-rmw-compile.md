# G short final-counter representation: positive local compiler screen

2026-11-01. Reviewer base `76ebb0f`. This is the new G Tiny/Tiny2 question,
not the closed standalone ec565/main47 termination experiment. No GPU execution,
official/service calls, production edits, benchmark or performance claim.

**The opt-in final-counter change produces a narrowly isolated SASS difference
without worsening selected-entry resources.** It passes this bounded compiler
screen. Formal source ordering, host launch eligibility, whole-score budget and
any eventual H200 validation remain separate gates; this is not release clearance.

## Exact inputs and selected builds

Control: root `muse_decode_layer/cand/g1b4_stage_release.py`,869076 bytes,
SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
This is owning607eb52bdaf24160, not a later packaged hybrid.

Opt-in prototype: owner `tools/g_short_final_rmw.py::build(source,enabled=True)`,
`.build/today-moe/short-final-rmw/on.py`,869551 bytes, SHA256
`d33915c7e8579d815a0adf8a182d685570185e8c6c1b5fd6c036854a5fbcdd44`.
Source guard restricts the new macro to unsplit WS Tiny, and only Tiny/Tiny2
receive `-DOKC_SHORT_FINAL_RELAXED=1`. The separate WGS fc RMW is unchanged.
The source owner and independent auditor own the PTX release-pattern proof;
this report does not infer safety from the compiler result or tests.

`tools/proj_g_short_rmw_compile.py` compiles actual `okc::moe_fused<false>`
for the two selected short build keys. Both use original runtime common
NS3/NSF3/FNST4/PRMTSC1/DQ7=1 flags plus actual Tiny/Tiny2 definitions; Tiny
also receives its actual FINPF1. Tiny has WSBLK2; Tiny2 has WSBLK1/CPF0.
No cap, geometry, math flag, family selector or boolean-variant sweep was used.
The existing source-factory ledger independently links these flags to actual
607e-family requests; the source owner independently checked this new manifest.

Established `local_cuda128_compile.py` driver is reused with only hardcoded
MAIN flags replaced by the actual short flags and expressions restricted to
the actual PDL-false entry. Device extraction preserves the original prefix
before Torch host extension, virtual fixed-width/type-trait headers and the
existing declaration-only `wgs_cta_of` adapter. It does not compile or prove
host launch wrappers, captures or whole-grid admission. NVRTC/PTXAS12.8.93,
compute_90a/sm_90a, C++17/O3 and existing nvdisasm12.8.90 were reused; no
new toolchain or GPU driver execution.

## Actual resources and lowered change

| Selected entry | Control | Opt-in |
|---|---|---|
|Tiny registers / barriers|147 /3|147 /3|
|Tiny stack / spill stores / spill loads, bytes|0 /0 /0|0 /0 /0|
|Tiny static instructions including alignment NOPs|5272|5272|
|Tiny2 registers / barriers|147 /3|147 /3|
|Tiny2 stack / spill stores / spill loads, bytes|0 /0 /0|0 /0 /0|
|Tiny2 static instructions including alignment NOPs|5328|5320|

Launch bounds remain288 threads/min1 CTA. These are compiler attributes,
not an H200 active-block or occupancy measurement.

For **each pair**, emitted PTX is byte-identical after dropping only `.loc`
and `.file` debug-location directives and replacing exactly one instruction:

```ptx
atom.acq_rel.gpu.global.add.s32 ...
// becomes
atom.relaxed.gpu.global.add.u32 ...
```

The signedness change preserves the32-bit add bit pattern; selected counter
values are nonnegative grid counts, not arbitrary signed overflow arithmetic.
Explicit writer `membar.gl`, WS_JOIN/CTA barriers, reader
`ld.acquire.gpu.global.b32`, math and every other PTX operand remain unchanged.
The compiler retains a strong GPU-scoped atomic, not a weak store or CTA-only
operation.

After branch target relocation normalization, the entire SASS sequence is
identical except one atomic-lowering block and trailing alignment NOPs:

```text
@P0 MEMBAR.ALL.CTA
@P0 MEMBAR.ALL.GPU
@P0 ERRBAR
@P0 CGAERRBAR
@P0 ATOMG.E.ADD.S32.STRONG.GPU ...
@P0 CCTL.IVALL
```

becomes the single `@P0 ATOMG.E.ADD.STRONG.GPU ...`. The extra ordering/cache
instructions came from lowering the acq_rel RMW. Their absence does **not**
mean an explicit source writer fence or reader acquire was removed: complete
PTX equality except that RMW proves those operations remain. Tiny adds five
tail NOPs after removing five executable instructions; Tiny2 has three fewer
tail NOPs. Static totals must not be translated to a runtime speedup.

The raw math/fence/barrier sequence comparator is intentionally false because
it includes these atomic-derived MEMBARs. The stronger scoped proof is:
all other PTX is exact, and all other SASS operands are exact after relocation
and NOP normalization. `proj_g_short_rmw_compare.py` asserts that exact known
six-instruction block is the only executable SASS change; it does not ignore
all fences or accept arbitrary instruction-count equality.

## Evidence and next gate

`evidence/2026-11-01-g-short-final-rmw-compile/` contains per-build manifests,
compiler logs, normalized comparison/hunks and gzipped raw PTX/SASS evidence.
Gzip is evidence storage only; no encoded source or binary goes in an entrant.
Full local CUBINs remain `/root/.cache/okc-g-short-rmw/`. Two selected keys,
one actual entry each were compiled, not every G kernel. Other family isolation
rests on the owner's exact source/macro guard, not an all-family compile claim.

No runtime saving, counter cost, exposed synchronization time or full-score
gain is measured. Passing the source proof and this compiler screen permits
the coordinator to assess a controlled short-only hypothesis and preregister
its whole-six-load budget; it does not authorize a harness/job here. Preserve
all BF16 rounds, cooperative progress, PDL-off policy, diagnostics and explicit
publication/acquire chains. Stop if independent proof fails, resources worsen,
or any later measured gain fails the full-score/null/regression gates. Never
broaden this into a relaxed-atomic sweep across WGS or other joins.
