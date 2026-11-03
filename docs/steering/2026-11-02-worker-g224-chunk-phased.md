# Worker: fixed G2/24 chunk-phased gather — resource STOP

November 2, 2026. Task started08:13:40UTC; source/CPU/four-entry compile completed08:22:24UTC, within the30-minute allocation. Authorization: captain d554937a0a51c1a903fa5546a8959fd9ecab1d10. Historical corrected-parent resource decision5798c533d23272e3583efeb10f215d3980c9075f and owning04e08381919c1c1be9b54326f80101bbd71418c8 remain unchanged. Captain owns README/shared ledgers; this report and its uniquely named tools/evidence are the only publication scope.

## Decision

**STOP_FIXED_CHUNK_PHASED_RESOURCES.** The one specified gather-lifetime treatment compiles for all four selected entries but retains nonzero stack and spills in every entry. This fails the unchanged registered zero-stack/zero-spill resource gate. No second native code change, phase-B repair, cap/ring/batch/compiler sweep, packaging, disassembly, GPU/driver query, challenge CLI, target check or production/default action follows. This is an offline resource discriminator, not a measured slowdown, speedup or rejection of all G2/two-CTA designs. Zero new performance-qualified winners.

## Actual artifact and exact source boundary

- [Runnable generator](../../tools/g_expert_g224_chunk_phased.py), [actual-generated source/CPU contracts](../../tools/test_g_expert_g224_chunk_phased.py), [fixed compiler entry](../../tools/g_expert_g224_chunk_phased_compile.py).
- Frozen parent generator SHA256114ed80ae87b485ade24a6db2dab6fbab7ab8ff2c4cbc512ca58c4bb7f799e1a; its public607 source remains ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. Parent selected native ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f remains the exact failed control, not overwritten.
- New actual selected native **53d09576f86e93b9baaa0145dcbdf1b60d8736743677d0a4de60e89864d18492**. Both captured native families use those exact bytes. [Selected source](evidence/2026-11-02-worker-g224-chunk-phased/selected.cu.gz), [complete native delta](evidence/2026-11-02-worker-g224-chunk-phased/gather.diff), [generated manifest](evidence/2026-11-02-worker-g224-chunk-phased/generated-manifest.json).
- With only wgs_gather replaced by the same marker, control and treatment have identical SHA2564de8a695f9a5ba6405c8d24f97e82b63ab0742d5f9f685ca99f818056c143fe4. Replacement reverses byte-for-byte to the control. Every native byte outside that function is unchanged, including geometry, caps, math, scratch/rings, launch, lifetime/publication and host paths.
- Append-only variant d67dc8d1c32ad512048c2998229448a97aee3d7d530c3bb2bd6b4accfdc9dd05,1028254B. The entire original1023788B parent Python prefix remains unchanged; appended code checks the control hash and replaces only the exact old gather body before checking the new native hash. No payload optimization is attempted. The variant remains above1MB and NOT targetREADY.

The sole treatment makes chunk0..1 the outer loop around the full q0..q1 batch sweep, with pragma unroll1 and j=0 inside each phase. GB stays8. Each batch has v[GB] and d[GB], loads eight rows for one chunk, then stores those eight rows. column=tid+chunk*NT and every source/destination bit-copy formula are unchanged. The original single global proxy fence, CTA_SYNC and tid0 padded-row release block remain verbatim after both complete phases. No early release or extra buffer exists.

Source-dimensional batch storage changes from64 payload+32 pointer scalar32bit slots to32+16; this is not proof of actual emitted overlap/liveness. The schedule relies on the same immutable p.x/routing inputs and disjoint xA chunk destinations. Our executable copy-mapping evidence supports the changed order without weakening the original publication boundary; it is not a full GPU memory/progress proof.

## Actual fixed four-entry compiler result

[Raw compile receipt](evidence/2026-11-02-worker-g224-chunk-phased/compile-result.json), [B4 log](evidence/2026-11-02-worker-g224-chunk-phased/B4-ptxas.log), [MAIN log](evidence/2026-11-02-worker-g224-chunk-phased/MAIN-ptxas.log). Compiler resource bytes below are static assembler reports, not executed traffic or elapsed time.

|Selection|Registers|Barriers|Stack B|Spill stores B|Spill loads B|Gate|
|---|---:|---:|---:|---:|---:|---|
|B4 false|96|16|8|8|24|STOP|
|B4 true|96|16|8|8|24|STOP|
|MAIN false|96|16|8|8|8|STOP|
|MAIN true|96|16|8|8|8|STOP|

The exact parent B4 rows were208/384/280 stack/store/load bytes; MAIN224/364/260. This one source-lifetime change materially reduces these static reports but does not satisfy zero-spill qualification and establishes no speed gain. No new disassembly/local-PC attribution was performed. In particular, the residual spills are **not localized here** to phase-B merely because old-binary attribution found phase-B spills.

Both families use the same fixed NVRTC13/PTXAS13 recipe and existing attested same-Smem adapter. Existing driver SHA256e695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c; adapted-driver SHA25675d651eaa373dd8b1eaa033f8793e4b7b64e0e7ef812109309370dd9364cc8fa. Each current NVRTC options list is exactly equal to its corresponding parent list: compute_90a/C++17/lineinfo and NS3/FNST4/NSF3/NW8/FAST1=0/EPI4=0/WGS1/SC6W1/PRMTSC1/DQ7=1, plus B4FIN1 only for B4. PTXAS13.0.88 remains sm_90a,-O3,--warn-on-spills. Actual captured original compiler flags and cpp_sources are unchanged; actual selected source is checked at the real factory entry.

NVRTC and PTXAS exit0 for both families. Both bools emit maxntid320,1,1 and minnctapersm2. The five emitted layout constants still pass: wc1248,wcp1248,wcpB2272,sizeof(wc)1024,sizeof(Smem)23096. Native layout assertions and dynamic request115328 are unchanged. The rounded-register necessary model still reports2CTAs for96, not a waiver for spills or target occupancy proof. Static/system reserved shared interpretation remains the prior unresolved gap. There is no host NVCC compile/link, target residency or CUDA API query. Established virtual type_traits/stdint and declaration-only wgs_cta_of adaptation remain disclosed; secondary device-prefix compilation is not target-toolchain equivalence.

## Executable CPU coverage and limits

Command: existing WSL host-validate Python, unittest discover -s tools -p test_g_expert_g224_chunk_phased.py -v. **4/4 PASS in20.528s**; [raw receipt](evidence/2026-11-02-worker-g224-chunk-phased/cpu-tests.txt). One initial CPU parser regex typo was corrected before the passing run; it never changed the native treatment or compiler options.

The tests bind to the actual generated body, not a separate proposed gather. They extract its q/row/count/tile/column/destination/source-permutation expressions and replay the fixed batch/chunk loops. All counts1..24, all256 compute tids, both chunks, partial batches, ranges starting/ending inside jobs or padding, multiple jobs, zero-length ranges and zero jobs are covered. Valid rows copy4096 BF16 positions exactly once; padding produces no stores while publication counts the original padded q range exactly once. Actual control/treatment copy dictionaries and release counts agree. Invalid missing-j-reset behavior is executed and rejected when the second phase encounters a stale later job; source negatives also reject changed width/columns/padding predicate, outer unrolling, CTA synchronization and release multiplier.

A real-entry CPU stand-in executes the original selected factory plus the append-only treatment. It checks selected MAIN/B4 sources, unchanged cpp_sources/flags and byte-identical short-path factory calls. Exact source reversal and outside-gather identity enforce the remaining fixed geometry/math/fence contract. CPU formula replay is not CUDA numerical execution, pointer-race/NaN proof or full scheduler proof.

## Evidence, attribution and stopping scope

[Manifest](evidence/2026-11-02-worker-g224-chunk-phased/manifest.json) pins owned code/report, compact new source delta, full current raw logs/manifests and losslessly compressed current PTX/cubins. [Verification](evidence/2026-11-02-worker-g224-chunk-phased/verification.json) checks all current compressed binary hashes against their owning compiler manifests and exact option-list equality to the parent. No duplicate old archive or source disassembly is included.

Researcher's corrected afeaa69078160680ad450c5b381cd73325a728aa report, [old-binary spill attribution](../../archive/docs/steering/2026-11-02-g224-spill-attribution.md), supports targeting gather but also records residual old phase-B pressure. Its76/99B4 and79/88MAIN gather site counts are static old-binary facts, not measurements of the new treatment. Initial misbound-bool d06b parser classifications are superseded. The present result answers only whether this one fixed chunk-phased schedule clears the registered resource gate: it does not.

Full pipeline inverse/range/split-finisher, ring-reuse/last-reader/replay/progress proof remains incomplete from the parent. Target numerical split grouping, fresh/replay correctness, actual cooperative admission/occupancy, host NVCC/link, prepare/payload limits and timing remain NOT_RUN/unqualified. No performance candidate, speedup, full-gap closure or target-ready claim follows from the CPU pass or lower static spill report. This exact treatment stops; no additional source repair, harness, packaging or target test is proposed under this allocation. Captain receives a queued handoff, never steering.
