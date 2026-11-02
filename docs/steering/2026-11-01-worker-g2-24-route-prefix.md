# Worker: corrected-storage G2/24 — executable source, compiler-proof STOP

Latest status: the dated addendum below records the completed four-entry offline matrix and supersedes the initial tooling-only STOP with an actual fixed stack/spill resource STOP. Historical evidence remains immutable.

November 1, 2026 Pacific / November 2 UTC. Allocation received 07:10:47 UTC; executable generator and five CPU contracts completed before the 07:40:48 checkpoint. Authorization: captain 5f904c445e73045beec58aa4d6d583fb8263e319, retaining original geometry allocation 206ccfbc82cb15066f12e66b21cccee2cf82308b. The old fixed-layout STOP 73edd656a52836ca85071bf5cb990b41e45387e2 and its archive are unchanged. No GPU, CUDA device/driver query, challenge CLI, quota query, judge dispatch, installation, subagent or production edit occurred.

## Decision

**STOP: the required native constant-offset proof does not compile under the fixed installed secondary NVRTC13 path.** This is a compiler/proof-instrument limitation, not a demonstrated layout/admission contradiction and not a performance result. I stopped without removing the required assertions, changing compiler/geometry/rings or emitting a measurement harness. The artifact is an executable scratch generator with partial executable contracts, **not a complete compiled/qualified candidate**. Zero new performance winners.

[Generator](../../tools/g_expert_g224_prefix.py), [CPU contracts](../../tools/test_g_expert_g224_prefix.py), [fixed compiler entry](../../tools/g_expert_g224_prefix_compile.py). Frozen evidence is [summary.json](evidence/2026-11-01-worker-g2-24-route-prefix/summary.json); [selected.cu](evidence/2026-11-01-worker-g2-24-route-prefix/selected.cu) and [native.diff](evidence/2026-11-01-worker-g2-24-route-prefix/native.diff) are actual generated artifacts, not an explanatory model.

## Pins and actual entry assembly

- Public607 base: muse_decode_layer/cand/g1b4_stage_release.py, SHA256 ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. Selected first MAIN literal c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71; the later alternate literal is not substituted.
- Generated selected native: **ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f**, 149352 bytes. Generated append-only Python variant: **bd22962e25f94abc3c96aa31ebd594f08c63ec7881252dba12ca37590ab21b93**, 1023788 bytes. The original 869076-byte prefix is unchanged. This uncompressed audit variant is above a future 1MB payload budget; no upload packaging was attempted or authorized.
- CPU capture executes the original actual _fork_native_moe selector, original _main_defines, and original/new _ma_ext factories for T1/8/16/32/128/256/384 with the selected PRMTSC/SC6 state. It captures two changed build keys, MAIN and B4; short Tiny/Tiny2/Mid CUDA sources, CPP, functions and flags compare unchanged. No full top-level preparation or GPU execution is claimed.
- Selected flags retain -O3, compute_90a/sm_90a, -lineinfo, NS3/FNST4/NSF3, FAST1=0/EPI4=0/WGS=1, SC6W1/PRMTSC1/DQ7=1; B4 adds B4FIN1. Only the allocated NW16->NW8 geometry define changes. Full actual flags are in [compile-result.json](evidence/2026-11-01-worker-g2-24-route-prefix/compile-result.json).

## What is implemented

The generator changes MAIN/B4 to 256 compute threads plus two helper warps, intended launch_bounds320,2, 128 weight rows/unit, FT_T24, three weight plus three X stages and cooperative 2*SM grid (H200 maximum264). It prunes unreachable NTT4..8 specializations and switches; selected reachable NTT1..3 retain RS decode, SC6/KPERM3 offsets and GS2 numerical operators.

The gather executes two 16-byte chunks per compute thread. A/B unit shifts become6/5; r16 becomes ub*8+wg*4+wp. Weight addressing maps new unit to oldunit=ub>>1, half=ub&1 and preserves both original WG pieces in each half. Four A units times two WGs publish the existing eight B-ready contributions to parent ub>>2. Fixups change to two WGs,265 slots; JMAX168, MF_XRS168, MF_DA512/MF_INTS6144, doneAs capacity168*16, and padded X rows4416 are propagated to native and Python scratch construction. Cost boundaries widen total*CTA to64 bits before division, and all stage/job shifts use10-pb.

The original final reduction/output mapping also depended on512 compute threads; a NW/grid-only patch would be incomplete. MAIN now runs two logical256-thread slices, preserving the original16-warp sum order and full4096-column coverage. B4 computes both original logical warp halves with8 physical warps, and writes512-column segments rather than leaving half of each1024-column segment unwritten. No epilogue precision boundary or norm/gamma operator was intentionally changed. Changed stream-K split grouping remains a separate future numerical obligation; source identity of operators does not imply bitwise equivalent full output.

All weight/X proxy fences, release/acquire operations, WGMMA retirement before X-slot release,128-thread WG barriers and compute/all-participant named barriers are retained. The original host occupancy check now requires resident>=2 and exact2*SM grid; that check has **not run**. Production files and incumbent dispatcher are unchanged.

## Union lifetime and layout obligation

Only wc[NW][32] and wcp[JMAX+1] are overlaid in an anonymous union. All other arrays and operand rings remain separate. To avoid relying on concurrent first activation of an array member, one compute thread performs S.wc[0][0]=0, then the compute CTA barrier precedes parallel routing writes. After route's final CTA barrier, threadfence_block and the existing barrier7 handoff, one X-producer lane performs S.wcp[0]=0; a warp synchronization precedes parallel prefix writes. This adds one explicit compute barrier at routing start and one producer warp synchronization; these are real source overhead, not a measured delay or free optimization.

C++ [class.union.general]/5 permits built-in assignment through an array-member/subscript expression to start the lifetime of a trivial union array member, ending the prior active member. The relevant primary draft is https://eel.is/c++draft/class.union.general . The implementation does not read an inactive array or retain a wc pointer across transition. Source checks reject late wc access, missing barrier7, missing first-activation compute barrier and missing prefix-transition warp synchronization. WGS early_issue returns and there is only one wgs_ranges caller. This is a source reasoning obligation, not a GPU concurrency result.

Native assertions require sizeof(Smem)==23096, union-member equal offsets, wcpB separation, aligned ring start23168 and dynamic request115328. These sizeof/alignment/dynamic assertions produced no diagnostics in the failed translation unit, but **the unit did not compile**, so there is no successful native-layout/build qualification.

## Exact compiler failure, not resource evidence

The reused fixed adapter strips only the torch host boundary, supplies its existing virtual type_traits/stdint headers and declaration-only wgs_cta_of forward declaration. It requests both actual bool entry expressions at compute90a; PTXAS13 would use sm90a/O3. Real host NVCC/extension compile/link is NOT_RUN; no usable g++/clang++/nvcc was found on the inspected WSL PATH. This does not establish no compiler exists elsewhere.

There were four bounded compiler invocations, retained without concealing the earlier failures:

1. Initial invocation lacked the existing library directory in LD_LIBRARY_PATH: NVRTC could not open its already installed builtins library. Nothing was installed. [Loader log](evidence/2026-11-01-worker-g2-24-route-prefix/attempt1-loader.log).
2. With the library path supplied, the direct offsetof assertions failed because offsetof was undeclared in the established virtual-header path. [Log](evidence/2026-11-01-worker-g2-24-route-prefix/attempt2-undeclared-offset.log).
3. The direct compiler-builtin spelling __builtin_offsetof was unsupported by this NVRTC frontend. [Log](evidence/2026-11-01-worker-g2-24-route-prefix/attempt3-builtin-offset.log).
4. The final source uses the installed official CCCL NVRTC offsetof representation (pointer difference from a null-base member) under RTC, and explicit stddef.h/offsetof for a future ordinary host path. NVRTC rejects this representation in static_assert with four **expression must have a constant value** diagnostics at generated lines257/258. [Final log](evidence/2026-11-01-worker-g2-24-route-prefix/nvrtc-final.log). Installed official header types.h lines26..31, SHA25673a074d0144f69c21b081d7e860cfb4acbe3adf8aa1cb72763c84d8c086586f9, pins the source of that representation. It is not evidence the expression works as an NVRTC constant expression.

Final source remains deliberately blocked at the mandated proof rather than conditionally skipping it. The NVRTC library SHA256 is a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11; builtins SHA25691dcd944d01da9c0f08fff5d779db136a47f6d62bdb63bae900d2f481e92c3a2. Installed PTXAS reports13.0.88 but **was never reached for this final source**.

|Selected requested entry|Compiler outcome|Registers / spills / stack / static SMEM|
|---|---|---|
|B4 false|NVRTC constant-offset error; no PTX|UNAVAILABLE|
|B4 true|Same source parsing gate; no PTX|UNAVAILABLE|
|MAIN false|Not attempted after STOP|UNAVAILABLE|
|MAIN true|Not attempted after STOP|UNAVAILABLE|

There are no emitted-resource rows, cubins, SASS measurements or WGMMA serialization results. In particular, neither old96 registers nor a new102-register budget is an observed resource result. Target reserved shared memory, actual occupancy and numerical/runtime behavior remain NOT_RUN. The captain's115328+reserve model is retained as conditional accounting, not qualification.

## CPU evidence and limits

Final command: python -m unittest discover -s tools -p test_g_expert_g224_prefix.py -v. **5/5 pass,11.533s**; [raw receipt](evidence/2026-11-01-worker-g2-24-route-prefix/cpu-tests.txt).

The executable contracts check actual generated helper expressions for both SC6/non-SC6 packed-piece bijections across both phases/all32 experts/units/stages/WGs, generated gather addressing for every legal1..24-token job row, both complete final-output mappings, original selected-factory assembly and scratch lengths, legal distinct-topk empty/single/full/spread/tail/general384-token job bounds, and all8 A-to-B publication contributors. Negatives reject old unit geometry, old slot/flag sizes, late alias reads, missing transition barriers and illegal duplicate top-k assumptions.

**Incomplete before STOP:** a full executable inverse/range/split-finisher progress scheduler, full ring-reuse/last-reader event model, replay initialization simulation and all their negative cases are not implemented. Retaining source fences and checking a few event-order strings does not substitute for those proofs. The tests therefore are not the complete CPU/progress gate required by the allocation, and are not CUDA numerical/NaN/output correctness, native compilation or a speed measurement.

## Handoff / stopping scope

No more implementation, compile spelling sweep, toolchain installation, target job or shape/ring/header rescue is pending in this allocation. A future decision would need a captain-authorized, authoritative way to satisfy the exact native constant-offset obligation and completion of the outstanding progress contracts; this report does not waive either. Current source/toolchain pairing is STOP, not all corrected-storage/two-CTA designs impossible.

Only after all offline obligations pass and a separate explicit target allocation: one ordinary candidate check using existing prepare/official-reference/profile flow can test actual2*SM cooperative admission, legal distinct routes, empty/fresh/replay outputs and numerical split grouping, retaining compact resources before failure. No graph-query/four-arm harness is proposed. Even a future successful ordinary check would give descriptive timing, not a causal gain against a different historical H200 node. Current stage measures no performance improvement.

README/shared ledgers remain captain-owned. This report, three uniquely named tools and the new evidence prefix are the only owned publication paths; old STOP archives and other sessions' WIP are preserved. Evidence file hashes are pinned by manifest.json and the publication Git commit.

## Addendum: November 2, 2026 07:59 UTC — fixed four-entry resource STOP

This addendum supersedes the initial tool-only stopping decision, not its historical evidence at746f93791049350880786bb4b184b433152d21ab. Captain explicitly authorized a bounded same-Smem NVRTC layout-probe adapter, then completion of the remaining fixed MAIN pair after B4 failed. No ring/header/geometry/math/compiler-flag rescue occurred. Actual selected CUDA remains ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f; the original native assertions remain unchanged. Original production/public607, short selections, attention and secondary sources remain untouched.

**Final decision: STOP_FIXED_RESOURCE_STACK_SPILLS, both MAIN and B4, both bools.** This is an actual secondary emitted-resource failure of this fixed implementation against the registered zero-stack/zero-spill gate, not the earlier unsupported offsetof witness and not a measured speed or target runtime failure. No target-qualified candidate or performance winner exists. No more compilation, packaging, tuning, GPU/CLI/query/harness work is pending.

### Same-Smem layout proof and adapter boundary

[Adapter](../../tools/g_expert_g224_prefix_nvrtc_layout.py) replaces only the two nonconstant RTC offset assertions in the compiler adapter, retaining actual CUDA/NVCC assertions. It appends five noinline device layout probes and an unlaunched adapter-only global anchor. Plain constant-return probes were optimized into the anchor despite noinline; the final probes use volatile inline PTX mov with an immediate constraint on the actual Smem expression. The returned PTX must contain exactly mov-literal, parameter-store and return in each probe; every value is checked before PTXAS. Anchor/probe functions are separate from the selected expert entry points and never launched. The adapter exact-reversal assertion rejects other device-prefix changes. Established virtual type_traits/stdint headers and declaration-only wgs_cta_of adapter remain disclosed.

Both selected families emit and pass the same constants: wc1248, wcp1248, wcpB2272, sizeof(wc)1024, sizeof(Smem)23096. Thus wc/wcp share an address and the separate B prefix starts after their1024-byte extent. Existing sizeof, ringstart23168 and dynamic115328 static assertions also compile. This establishes this secondary frontend's actual Smem layout; it is not host NVCC compilation, union-concurrency/progress proof, numerical equivalence or target occupancy.

[Full frozen compiler evidence](evidence/2026-11-01-worker-g224-prefix-resource-stop/manifest.json) includes both manifests, full PTX and cubins compressed losslessly with decompressed byte hashes, raw NVRTC/PTXAS logs, compiler entry receipts, readelf sections and executed B4 runner snapshot. The exact existing secondary driver is archived too. The first B4 anchor/parser development failures are retained as layout-anchor.log and layout-asm.log; the successful layout/resource attempts are layout-final.log and MAIN-layout-final.log. These are offline invocations, not challenge jobs.

### Complete fixed resource matrix

|Selected entry|Registers|Barriers|Stack B|Spill stores B|Spill loads B|Decision|
|---|---:|---:|---:|---:|---:|---|
|B4 false|96|16|208|384|280|STOP: stack/spills|
|B4 true|96|16|208|384|280|STOP: stack/spills|
|MAIN false|96|16|224|364|260|STOP: stack/spills|
|MAIN true|96|16|224|364|260|STOP: stack/spills|

All four entries emit maxntid320,1,1 and minnctapersm2. NVRTC and PTXAS exit0 for each family; the runner then deliberately exits1 on emitted stack/spills. PTXAS13.0.88 uses sm_90a, -O3, --warn-on-spills. Both NVRTC manifests retain compute_90a, C++17, lineinfo, NW8/NS3/FNST4/NSF3/WGS1/FAST1=0/EPI4=0/SC6W1/PRMTSC1/DQ7=1; the only family define difference is B4FIN1. Both bool instantiations are the actual selected moe_fused templates, not dormant alternatives. PTXAS reports zero C7513 serialization warnings in both receipts; absence of that warning is not progress or overlap proof.

For compiled shared storage, PTX contains only the external dynamic shared declaration, no explicit static shared declaration. Each expert cubin .nv.shared section is1024B (readelf size000400); the unlaunched anchor has the same section size. This raw ELF fact does not establish whether that1024 is additional user shared storage or compiler/system reservation. No runtime sharedSizeBytes/reservedSharedMemPerBlock property was queried, and no overall two-CTA residency PASS is claimed. Dynamic request remains115328. The prior116352/block,232704/pair,768-margin calculation is only the conditional zero-additional-static plus1024-reserve model. The spill failure is decisive independent of this unresolved shared accounting.

Captain's rounded register correction is enforced in the actual runner and executable negatives:320threads=10warps;256-register/warp rounding; four SMSPs with16384registers each. At R96,3072regs/warp permits5warps/SMSP,20/SM,2CTAs. R97..102 rounds3328, permits4warps/SMSP,16/SM,1CTA. Unrounded65536/(2*320)=102 is not the gate. This is a necessary header-equation CPU calculation under SM90 assumptions, not target occupancy. Installed header pin remains163b8f296ef2a33c799e2dc8c525b37166da9217bcc6a3c1d26b1c10aca5e22b. R96 does not waive stack/spill rejection.

### CPU result, incomplete obligations and scope

Combined source plus offline-receipt suite: **9/9 PASS in8.744s**, existing WSL host-validate Python; [receipt](evidence/2026-11-01-worker-g224-prefix-resource-stop/cpu-tests.txt). New executable tests replay compressed PTX/cubin hashes, exact four-entry rows/flags/layout, missing/wrong layout probes, lost native assertion, nonoverlap mutations and actual runner equation R96 versus R97..102. The original five actual-generated formula/factory contracts pass unchanged. These checks validate evidence/rejection logic, not performance.

The previously listed full inverse/range/split-finisher scheduler, ring-last-reader event simulation and comprehensive replay/progress contracts remain incomplete: the decisive fixed resource STOP occurs first. They must not be relabeled PASS. Native host/NVCC compilation/link is NOT_RUN; no usable g++/clang++/nvcc was found in inspected WSL PATH, and no installation/detour occurred. Numerical split grouping, general runtime bounds/progress, actual cooperative admission and timing remain NOT_RUN.

Append-only variant remains bd22962e25f94abc3c96aa31ebd594f08c63ec7881252dba12ca37590ab21b93,1023788B, above the1MB upload budget; no packaging work is performed. This is not targetREADY even apart from spills. The generator remains a real unfinished/unqualified scratch artifact, not a candidate that passed the complete source/CPU/compiler gate. No claim that all G2/two-CTA pipelines are impossible follows.

There is **no next target test for this stopped fixed artifact**. Only a separately authorized future implementation that clears the full zero-spill/layout/progress/payload gates could justify the previously described minimal ordinary official-reference/profile qualification; no graph-query/four-arm harness is proposed. Current result changes the decision from tooling HOLD to fixed emitted-resource STOP, not to a GPU request. Captain owns README/shared closure.
