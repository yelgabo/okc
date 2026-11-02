# G2/24 local traffic: gather-dominated, not per-stage WGMMA spilling

2026-11-02, offline artifact attribution only. Owning current resource publication: `04e0838`; source/base ba9, generated native SHA256 `ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f`. Historical offset STOP `746f937` is superseded only on the attested secondary-layout/resource question. Captain allocation `5f904c4`. No production/source treatment, recompile, flags, GPU/driver/judge/CLI, install or occupancy query was performed. README/shared logs and worker artifacts remain untouched.

## Decision

**The zero-spill STOP has concrete conditional dynamic-traffic support, but is still a conservative performance gate, not proof this design loses.** Most spill instructions are in the activation gather's repeated eight-row batch, not routing/prefix/range/division or the repeated inner WGMMA stage pipeline. The remaining spills are phase-B segment state/epilogues and entry/drain. There is no evidence here of spills on every WGMMA stage, measured cache traffic, exposed latency, performance loss, or a correctness failure caused by spilling. Keep the exact implementation STOP; no rescue or gate relaxation is authorized.

## Exact binding and reproducible inventory

Inputs are the worker's archived B4/MAIN PTX, CUBIN, manifest and logs under `evidence/2026-11-01-worker-g224-prefix-resource-stop/`, and `evidence/2026-11-01-worker-g2-24-route-prefix/selected.cu`. The CPU analysis reconstructs the attested NVRTC device-only adaptation without executing worker/compiler code: two offset assertion texts removed, five layout probes inserted, existing declaration-only `wgs_cta_of` inserted. Its complete device hash matches **both owning manifests**. This resolves the one-line displacement after the declaration; the inserted probes are after the expert body. Source/native versus adapted line numbers are kept separately.

B4 CUBIN SHA256 `dbcc51bcb6b586430755fc13bc477ec6f0803a09968ec7ea8947eaa0f660ce7f`; MAIN identity is in the owning manifest and the copied [disassembly receipt](evidence/2026-11-02-g224-spill-attribution/disassembly-receipt.json). Both decompressed CUBIN hashes and PTX hashes were checked against their manifests. Installed nvdisasm13.0.85 was run with source/inline debug, PTX debug, and basic-block CFG+PC output; GNU readelf supplied sections. Actual binary paths, versions, SHA256 and commands are in that receipt. This does not strengthen secondary NVRTC13/PTXAS13 into actual target NVCC equivalence.

[Every local access, entry+PC+opcode+source+PTX binding](evidence/2026-11-02-g224-spill-attribution/local-accesses.tsv) and [complete attribution/natural-loop records](evidence/2026-11-02-g224-spill-attribution/attribution.json) cover both `moe_fused<false>` and `<true>`. PC is relative to each named `.text` entry. All local-access PCs map into reachable emitted CFG blocks. Loop membership uses dominator-backed natural backedges and predecessor closure, not a contiguous PC interval: some loop blocks are placed far away. Source/debug disassemblies and CFGs are compressed beside the receipts. Scripts only reconstruct/read/disassemble; they do not compile or run a CUBIN.

## Counts and control-flow classification

The following results are identical for the two bool entries within each family:

| Family | Gather batch local instructions | Phase-B outer segment local instructions | Phase-B entry/drain local instructions | Total |
| --- | ---: | ---: | ---: | ---: |
| B4 | 76 | 18 | 5 | 99 |
| MAIN | 79 | 7 | 2 | 88 |

These are static instruction-site counts, not executions. All opcodes are scalar/64-bit LDL/STL variants. Their **static operand-width sums** exactly reproduce the assembler's spill rows: B4 384 store/280 load bytes; MAIN 364/260. Instruction encoding occupied by those local sites is separately1584B/1408B (16B per instruction). Neither number is executed traffic, unique stack footprint, HBM bytes, or timing. Owning stack sizes remain208B/224B, registers96, barriers16, launch320 threads/min2CTAs. The registered fixed96 ceiling is unchanged; no cap97..102 suggestion is made.

Gather alone accounts for B4 340 static store-width bytes/212 load-width bytes, MAIN344/224. B4 gather spills occupy PCs `0x4700..0x84c0`; MAIN `0x4700..0x84d0`, with every such access in natural loop `.L_x_111` (gather `qb += GB`). Examples in either bool:

- Both families `0x4700 STL.64 [R1+0x90], RZ`: pointer-state initialization in the gather batch. This instruction is unpredicated after entering the nonempty batch; it is not a one-time routing store.
- B4 `0x4be0 STL.64 [R1+0x98], R8`; MAIN `0x4bc0 STL.64 [R1+0x98], R4`: destination address preparation/preservation.
- B4 `0x7e50 LDL.LU R74,[R1+0x34]`; MAIN `0x7e50 LDL.LU R74,[R1+0x30]`: gather state reload, source native1911 guard, corresponding PTX branch34894 in MAIN/36164 in B4.
- Gather tail reloads map to native1927 `if (d[k][chunk]) *d[k][chunk] = v[k][chunk]` after loading all batched input vectors at native1918. Full PC list is in the receipt.

MAIN `0x7e90 STL [R1+0xa0],R0` carries debug attribution to entry `threadIdx.x` (native2371), but **physically belongs to the gather batch loop**. Likewise two local instructions in each family carry a stale/inherited phase-A inline attribution while physically at phase-B setup/header. Do not infer execution phase from the last source annotation alone: the raw `role`/TSV class is an inline-debug hint, whereas `execution_region` records CFG+surrounding phase context.

Phase-B witnesses:

- B4 `0x17be0 LDL R3,[R1+0xc]`, header `.L_x_445`; MAIN `0x17e10 LDL R4,[R1+0x4]`, header `.L_x_359`. Both reload phase-B outer segment state. The immediately preceding STL.64 is outside that natural loop. Surrounding code loads `S.wgr[2:4]` for `wgs_phase<true>`; the inherited phase-A `wgs_split` annotation is not proof of phase-A execution.
- B4 `0x19730 STL [R1+0x8],R7` has `wg_fence`/first-stage inline attribution; `0x19830 LDL R66,[R1]` is deferred-flush state. Both lie in the **outer segment loop**, not the repeated inner stage loop.
- MAIN `0x19920 STL [R1],R20` maps through packed decode at native282/2163, first `stage(true)`2187; `0x1ca70 LDL.LU R7,[R1]` is the post-stage split decision2192. This is not a scalar-interleave or packed-RNE treatment; those pre-existing operators were not changed.
- B4 `0x1e2d0/0x23ac0/0x28c90 LDL.64 R6,[R1+0x10]` feed phase-B output/partial-sum epilogues (native2000/2205, NTT arms). MAIN `0x1d4b0/0x21f80/0x26770 LDL.64 R16,[R1+0x8]` similarly map to FP32 output stores1984/2205. These are branch-specific static sites, not three loads necessarily executed by every token.
- B4 `0x28ed0 LDL.LU R0,[R1]` is final deferred flush outside the segment loop; it is not dense layer finalization.

Each family has eight emitted HGMMA-containing natural loops per bool: three inner stage loops per phase and two outer phase/segment loops. **All six inner loops contain zero LDL/STL; phase-A outer loop also zero; only phase-B outer contains the local sites above.** All local accesses in all executable text sections are accounted for in the two expert entries; the layout anchor has none. There are no attributed local sites in the selected routing, prefix/ranges, 64-bit boundary/ownership division, weight/X-producer loops, or post-expert layer finalization. This is exact emitted-code attribution, not a claim those regions are inexpensive. Some optimized source locations are inherited/imprecise; the receipts explicitly retain these ambiguities rather than assigning a unique source variable to every spill slot.

## Explicit arrays versus assembler spills; dynamic meaning

The archived PTX contains **no `.local` declarations, `ld.local` or `st.local` operations**. Thus these SASS local accesses are assembler register-allocation spills, not explicit PTX-local array storage. Source arrays can nevertheless cause the pressure. Source `v[8][2]` holds16 uint4 payloads =256B/64 32-bit words; `d[8][2]` holds16 64-bit destination pointers =128B/32 words. The loads-first/stores-second gather exposes up to384B/96 words of raw payload+pointer state before loop/index/predicate state. This is a source live-state inventory, not a theorem that every word must occupy a distinct physical register: predication, address rematerialization and scheduling can change allocation. But the actual spilling is overwhelmingly in that exact region, while the six inner stage pipelines remain free of local instructions.

**Chosen pressure hypothesis:** retaining two chunks for eight rows until the second store loop exhausts the uniform96-register budget in gather; additional phase-B pending/range state survives across segment work. This is more specific than blaming routing's64-bit divide or repeated WGMMA accumulator spills. The present artifacts do not identify every allocator decision or prove gather is the sole pressure source.

Conditional dynamic evidence is stronger than an unused-code spill counter: entering a nonempty gather batch reaches actual unpredicated local stores, and the CFG repeats them for additional batches; phase-B segment local sites can recur with units. Exact batch/segment counts, active lanes, predicates, cache residency and exposed time are **unobserved**. No archive trace was used to assert their frequencies. Spill counts cannot establish a whole-six benefit, a traffic total, or that this implementation must be slower than a zero-spill alternative. The STOP remains a predeclared resource/performance gate, not a memory-order/numerical disqualification.

## Smallest warranted future discriminator, not authorization

If captain explicitly reopens source/compiler work, test **one gather-only lifetime-shortening schedule**: keep the same two chunks, GB8, payloads/layout, flags,320-thread/min2 geometry, three+three rings, all arithmetic/BF16 boundaries and publication/proxy fences, but consume each chunk's gathered values before retaining the next chunk's values. Compare the exact local-PC inventory/resources against these controls. The discriminator is whether the gather-local sites/stack disappear without simply migrating pressure into the stage pipeline or worsening resources. Persistent phase-B spills still fail the existing zero-spill gate; reducing gather spills alone does not qualify the design or earn GPU time. This report implements no such rescue and requests no gate relaxation.

Actual target NVCC/resources/occupancy/progress/numerics/timing remain NOT_RUN; payload1023788 above1MB remains separately NOT_READY. No primary-source web survey was needed or performed: scope was the specified existing artifacts, inspected Nov2~08:00-08:16UTC. Captain owns indexing and next allocation; this task ends at publication and a queued notification.
