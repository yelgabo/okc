# Fully timed WGS BF16 SiLU table: secondary-only survivor

November2,2026,10:34UTC; bounded start10:03:52UTC, deadline10:38:33UTC. Evidence checkout8689680, then8f3bb13; reconciled captain allocation8f69050, startup findingfd27770 and terminal reconciliationed6b465. ONE new timed design, not G224 rescue. Prior53d095/0d84, own e1271ad5e5e34b24 and untimed HOLD5fdbd56 remain intact. No GPU, CUDA driver, host link, challenge CLI, public/scored/default action, installation, compiler sweep or subagent.

## Decision and exact treatment

**Implementation complete; fixed secondary resource/codegen screen survives. HOLD before target/performance allocation.** No measured gain or performance-qualified candidate. [Exact physical LF candidate](evidence/2026-11-02-worker-timed-silu-table/generated/candidate.py):870087B, SHA256d70a1bcc6202e37562b675fbab3251cdbe1b6fd81569442b3a179270aebae401. Final MAIN CUDAf5344c9d8d82cae177050a8d4bfa80b3f5f88b9fcfb35f726f777d55ad22f355. Generator `tools/g_timed_silu_table.py`; tests `tools/test_g_timed_silu_table.py`.

Base working public607 is exact Pythonba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab/nativec7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71. Rationale: exact cached selected B4/MAIN NVRTC13 controls exist at `.build/worker-g224-reference/{B4,MAIN}`, avoiding additional common4cbb/G224 integration. No production/default/public file changes.

Initial scope frozen before treatment: WGS B4 T32, MAIN T33/128/256/384. T1/8/16 retain original helper/factories/timed native path and launch NO producer. Only already-rounded `wgs_epi_tile` gate nonlinear changes to ordinary readonly FP32 lookup, indexed by upper16 exact embedding bits. Gate/up scale, BF16 round/widen, final FP32 multiply and BF16 store remain. Four inactive MAIN nonlinear sites and all five independently inventoried helper sites stay unchanged. Selected emitted WGS windows are checked independently for both bools, not inferred from one literal.

Every valid selected timed native call launches256blocks*256threads, writes ALL65536 FP32 entries from exact BF16 embeddings using existing `SILU_DIV(gate, 1.f + __expf(-gate))`, on caller current stream immediately before unchanged checked cooperative expert launch. No PDL/capture query, route-value bypass, table cache/global setter or reuse across calls. Valid empty-local-route calls follow the same producer path; invalid M0 remains rejected by the original admission. Real graph replay execution is NOT RUN; source has an unconditional preceding timed producer, not startup-only work.

## Actual startup and ownership correction

Captainfd27770 correctly found first draft allocated only in production prepare, missing `_release_expert_self_test`'s independent `_ma_make_scratch`. Fixed within initial allocator/binding scope: factory allocates uninitialized65536-float table; production and private startup guard/copy only original three buffers, then append private table WITHOUT reading/filling its numeric content. Actual factory, production allocation statements and private-startup prefix execute in CPU mocks, checking independent state/table identities. Prepare performs no table math. Short device code remains unchanged.

Optional debug ABI is explicit: MAIN table is `scratch.back()`; normal four-buffer calls keep `p.dbg=nullptr`; optional five-buffer calls keep debug at index3 and table at index4. Selected DEBUG0 is unchanged. Helpers are byte-identical/DEBUG0; no claim about enabling another debug build. Final corrections change Python allocation/startup and native HOST binding only. Final device-prefix bytes are EXACTLY equal to both compiled treatments: [final linkage/codegen/ROI receipt](evidence/2026-11-02-worker-timed-silu-table/final-link-codegen-roi.json). No second device spelling or compile rescue. Compiled full CUDA and final full CUDA have distinct retained identities.

## CPU and secondary evidence

Seven CPU tests PASS, including13 mutation negatives: missing/conditional producer, static/global/wrong table binding, dropped BF16 round, lower precision, wrong index/domain, filled prepare allocation, wrong helper factory and private-startup guard misuse. Exact whole-file reverse proves unchanged content outside four intended edit groups; ordered Python comment tokens identical, preserving all notices/provenance. CPU classifies65536 words:65280finite,2inf,254NaN; checks finite embedding/index and signed zero. CUDA SiLU equality, NaN payload behavior and final BF16 numerical equivalence are NOT RUN.

Actual `_ma_ext`/`_fork_native_moe` capture: eight requested T values resolve to five production build keys (tiny/tiny2/mid/B4/MAIN). Historical seven-factory coverage also invokes the two existing short-RMW-ON build requests as unchanged-source controls; they are NOT selected production paths or independent candidates. All seven CPP/ordered-flags comparisons match; only B4/MAIN CUDA changes. No routing/flag choice added to production.

Fixed NVRTC13/PTXAS13 recipe, virtual headers, declaration-only adapter and actual ordered flags exactly match cached originals. First invocation failed before compilation opening existing builtins; archived initial loader receipt. Identical source reran with established `LD_LIBRARY_PATH=/root/.cache/okc-cuda130-88/nvidia/cu13/lib`; no package/compiler change. Device-prefix recipe omits host NVCC/link/runtime.

| Family, both bools | Registers | Stack B | Spill store/load B | Barriers | Gate |
|---|---:|---:|---:|---:|---|
|B4 original/treatment|96|16|120/216|16|all deltas0|
|MAIN original/treatment|96|8|20/52|16|all deltas0|
|Producer, both builds|10|0|0/0|0|no spills|

Expert bounds576threads/min1CTA unchanged. Existing consumer spills are NOT zero-spill qualification, only no-worse parity. Params adds one pointer; Smem, dynamic shared expressions, fences/barriers, geometry, scheduling/admission unchanged by exact source reversal. Actual host NVCC/occupancy/resident-grid behavior NOT RUN.

Producer PTX retains scalar mul(-log2e),ex2.approx,add1,div.approx then FP32 global store; no added FTZ modifier. SASS retains scalar FMUL/MUFU.EX2/FADD/MUFU.RCP/FMUL/STG, with compiler exponent-range handling. Each of four expert entries has108 emitted selected-site windows checked for gate/up BF16 conversions, exact bit index, `ld.global.nc.f32`, final scalar multiply and BF16 store. This replaces four nonlinear PTX operators, NOT up multiplication or entire epilogue. Annotated SASS/PTX excerpts, raw logs/manifests, compiled inputs and full-artifact hashes archived; full large PTX/cubin/SASS remain locally hash-pinned. Secondary compile/codegen is not numerical or speed proof.

## Cost, whole-six gate and handoff

Affected official loads: decode32,verify16,verify32 (3/6). WGS waits for accumulators then executes this epilogue on compute WG: dependency evidence, NOT quantified exposed time. Each affected invocation pays65536 nonlinear evaluations,262144B writes, one extra launch and dependent table reads. For R local route-tokens,4096*R FP32 lookup requests =16384*R logical bytes; actual traffic depends on caching. Cold-L2 behavior, launch/producer costs, dependent-load latency and dynamic routed counts remain unmeasured. No speed forecast from operation counts.

Recomputed frozen607 whole-six arithmetic with ONLY these three loads benefiting. Hypothetical1%/3%/5%/10% reductions of their ENTIRE expert durations yield0.2608%/0.7886%/1.3251%/2.7056% whole-layer gains. >=1% needs3.79184% uniform affected-expert reduction assuming zero replacement cost/full critical-path exposure. These are scenarios, not a credible measured SiLU budget. The existing >=1% repeatable opportunity/noise gate is NOT waived.

Captain owns independent rules/profitability/public-source analysis and README/shared ledgers. This bounded implementation allocation ends with source/secondary evidence only. Park if a credible >=1% whole-six opportunity cannot survive complete producer/launch/lookup costs. No automatic check. Future target work needs separately authorized CUDA all-domain/consumer numerics, actual startup/fresh/empty/replay, target compiler/resources/admission and complete matched performance comparison. No reader/four-arm/benchmark harness is created. No losing performance experiment occurred.

CPU reproduction: `python3 -m unittest discover -s tools -p test_g_timed_silu_table.py -v`. Exact generation: `python3 tools/g_timed_silu_table.py`. Fixed compile is opt-in `--compile-fixed`, using installed cached recipe only; not a request to repeat it. Publication commit returned separately; evidence manifest and all explicit publication files verified remotely after push. README/sharedlogs stay captain-owned per allocation.
