# Producer-decoded shared-A pipeline: fixed implementation ready for ordinary private check

November 2, 2026. Captain allocation **13f5e41f**, [shared-weight pipeline](../../../docs/steering/2026-11-02-driver-shared-weight-pipeline.md); pickup indexed at **e3e4b346**. Existing OKC Developer owns this implementation. **OK Driver remains sole judge dispatcher and README/index owner.** No worker judge action, public/scored submission, production-default change or measured saving is claimed.

The concrete candidate is [generated/candidate.py](../../../docs/steering/evidence/2026-11-02-worker-shared-a-pipeline/generated/candidate.py), ready for the captain's one allocated ordinary private full-layer qualification/measurement. Six focused CPU tests and all eight selected secondary device compilations pass. Target host compilation, actual cooperative admission, numerical/progress/startup/replay qualification and performance remain the ordinary check's gates. A previous measured loss remains parked; this changes weight-decode ownership and operand residence rather than tuning that candidate.

## Frozen source and selection

| Artifact | SHA256 / bytes |
|---|---|
| Complete immutable ba9 control, public job607eb52bdaf24160 | `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab` |
| Actual original primary native c7 | `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71` |
| Fixed shared-A native | `c2958fc997e11cb4670e108a0509976cd9ce1763a8af89d2012cfc0ff6cea2c9` |
| Full ordinary layer candidate | `fbca3b75125dd45b3e2801358e95aa14a16750860482989a5fef444a576cd578` / **894471 LF bytes** |

Generator `tools/wgs_shared_a_pipeline.py`; focused tests `tools/test_wgs_shared_a_pipeline.py`. Entire original prefix/notices are retained, followed by readable checked-anchor native transformation, lossless packers and selected factory/scratch overrides. No compressed/opaque source, persistent BF16 weight expansion or new general observer framework. The code rejects an unexpected base/native hash. Canonical native, diff, factory descriptors, compiler driver/adapters/logs and losslessly compressed PTX/CUBINs are pinned by the [archive manifest](../../../docs/steering/evidence/2026-11-02-worker-shared-a-pipeline/archive-manifest.json).

Only B4/T32 and MAIN/T128,T256 change in both layer contexts. The actual `_fork_native_moe` entry is exercised at1/8/16/32/128/256/384 for raw/SC6 states. First three factory calls compare exactly equal to the control, including native/CPP source, flags, exports and module names. Original512-consumer finalization is byte-identical for B4 and MAIN; no two-slice/two-WG finalizer is inherited.

## Concrete dataflow and costs

Keep256 output rows/unit, four compute warpgroups,64-token jobs, count-dependent NTT1..8, original UBL5A/4B and one CTA/SM. Fixed Kstage128 with two distinct compact raw, two decoded BF16 A and two X stages. Thread roles are512 compute,128 decoder/weight-TMA and32 X producer: **672 total**, not704. Module source/flags determine the build cache key; no process-global timed treatment switch.

The decoder leader prefetches two raw stages. All128 decoder threads read the ready raw stage, each visits the four old virtual-WG fragments, and applies the exact existing `dq4` plus `bmul2l/h` primitives. They write the resulting BF16 fragment bits into a separate shared matrix. Consumers issue descriptor/descriptor BF16 SS-WGMMA rather than maintaining register-A ping-pong decode in the issue loop. Decode remains timed; preparation only repacks original code/scale bytes losslessly.

Total arithmetic is not reduced. Each decoder thread takes roughly four old consumer threads' decode work. Each K128 stage additionally writes65536BF16 A bytes and logically reads that shared A through SS-WGMMA, plus new producer/consumer arrivals. Halved K payload doubles stage handshakes; the512-consumer grid/output ownership and64-token reuse avoid the previous candidate's doubled logical X requests. These are costs included in complete layer measurement, not assumed hidden time or bandwidth savings. Register/spill costs below remain in the fixed implementation without a sweep.

The captain's fully exposed conditional model needs about11.37% of affected expert activities to become net endpoint saving for3% layer gain, or15.28% with5us replacement cost/load. Those activities are not removable-time bounds and neither number predicts this pipeline's result. We fund a bounded measurement of dominant-path coverage, not a promised top100 gain.

## Layout, mathematics and publication contract

K128 packing uses full256-row units/four WG pieces. Each WG piece contains4096code bytes and512 raw scales, or384SC6 bytes. The K128 bijection keeps group16 scales with their codes; gate/up row ordering and activation gather/intermediate layout use the same permutation for both contractions. SC6 reconstructs original scale bytes before the same BF16 decode operations. Raw eligible scale bytes8..126 and all64 compressed code values are covered by CPU tests. Unchanged source retains global FP32 postscale, BF16 boundaries, SiLU/up order, expert normalization and slot-ordered weighted combine. GPU SS accumulation/numerical equivalence is still required; CPU fragment checks do not prove full epilogue equivalence or target results.

Producer store index in BF16 elements is `F(row,k)=(row>>3)*1024+(k>>3)*64+(row&7)*8+(k&7)`. Virtual thread `wp=t>>5,lane=t&31` writes the four old fragment pairs to `(row,k),(row+8,k),(row,k+8),(row+8,k+8)`, where `row=64*targetWG+16*wp+(lane>>2)` and `k=16*k16step+2*(lane&3)`. All32768 stage elements are covered once. No-swizzle K-major descriptors use WG base+16384B, LBO128B/SBO2048B, transA0/transB0, K16 encoded-address increment16. The SS instruction includes both descriptors and both transpose immediates. These follow the [PTX shared-memory layouts/strides](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout) and [WGMMA operand syntax](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma), retrieved November2.

The lifetime chain is explicit:

1. Raw TMA completion is observed before generic decoder reads. Raw-empty receives128 arrivals only after every decoder has consumed all four WG pieces. Leader can then refill the reused raw slot; named barrier8 joins only those128 decoder threads.
2. Every generic A writer executes `fence.proxy.async.shared::cta` before its A-full arrival. A-full requires128 arrivals, so consumers cannot observe a partial stage. Initial barrier creation/publication and all672-thread startup joining are retained.
3. All512 consumers observe A-full and X-full for the matching parity. After their final SS commit each WG executes `wg_wait<0>()` before its threads arrive on either A-empty or X-empty. Both empty barriers require512 arrivals; neither operand can be overwritten until all four WGs retire async reads. These requirements follow the [PTX async-proxy completion/fence rule](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-async-proxy). Test passing is not substituted for that chain.
4. Two-slot phases continue across both contractions in the same reordered stream as consumers. No out-of-range raw prefetch is issued at the tail, empty work issues none, and consumer terminal waits/flushes remain. Producer work does not add an oversubscribed inter-CTA barrier; actual one-CTA cooperative admission remains mandatory.
5. Each unit has32 K stages. Original256-row phase-A ownership emits128 inter columns/unit, so phase-B stage `st` waits on **doneAs[j*32+st] count4**; each A WG publishes to **index ub**, never `ub>>1`. Job/epoch resets, scratch capacity, partial completion/designated finisher and range coordinates all use32 stages. Pending work is flushed at reachable commit group ci3 before replacement and again at phase end. Inter-CTA release/acquire and original finalization remain.

Captain reported independent read-only acceptance of the earlier d654e330 native's protocol, conditional on CPU/target gates. Final c2958fc9 native has identical code after stripping comments; only descriptive comments changed after that review. The actual CPU packing gate caught and corrected a retained two-WG reshape in the draft Python packer before freeze. That draft was never dispatched. No register/unroll/tile/stage tuning occurred.

## CPU and compiler evidence

**6/6 focused CPU tests pass**; [output](../../../docs/steering/evidence/2026-11-02-worker-shared-a-pipeline/cpu-tests.txt). They cover actual selected factories and unchanged decode/finalization primitives; exhaustive A-store/SS-descriptor bijection and wrong-pitch negative; full256x4096 logical matrices over all32 K stages with raw/SC6 bit decoding and BF16 CPU products; raw scale-domain coverage; activation/gate-up/intermediate permutations; counter/retirement source invariants and skewed four-WG slot-reuse CPU model; and scratch/odd33/257/384/tail/range/split capacities. CPU models are not GPU memory-ordering, scheduling or speed evidence.

Both selected B4/MAIN bool entries compile for each supported SC6/raw factory branch: **eight actual entries**, same fixed geometry. Secondary toolchain is installed NVRTC13/PTXAS13, SM90a, disclosed virtual headers, C++20 and existing declaration-only forward adapter. Actual factories specify host and CUDA C++20, `-O3`, SM90a codegen/lineinfo and the recorded incumbent defines; no blanket fast-math addition. There is no local target NVCC/host build claim. All entries emit maxntid672/minnctapersm1, with **zero C7513 warnings**.

| Family, both bool entries; raw and SC6 | Registers | Barriers | Stack B | Spill store B | Spill load B |
|---|---:|---:|---:|---:|---:|
| Shared-A B4 |80|16|40|184|392|
| Shared-A MAIN |80|16|24|108|192|

For scoped comparison, previously archived same-C++20 c7 device controls report B4 96regs/16barriers/16stack/120store/216load and MAIN96/16/8/20/52, in [original reference evidence](../../../docs/steering/evidence/2026-11-02-worker-wgs-k128-full-reuse/reference/compile-result.json). Those controls were not rerun and are not target host measurements. Treatment spills increase; the lower register count does not establish a win or producer efficiency.

## Complete storage and runtime gate

Emitted constant-only layout witness gives `sizeof(Smem)=22240`, aligned header22272, JMAX88, fixup5246976floats and672threads. No extra header fields or raw/A aliasing.

| Storage | SC6 bytes | Raw bytes |
|---|---:|---:|
| Aligned header |22272|22272|
| Two compact raw stages |35840|36864|
| Two BF16 A stages |131072|131072|
| Two64x128BF16 X stages |32768|32768|
| **Dynamic total** |**221952**|**222976**|
| Illustrative total including1024 reserved/block |222976|224000|

The illustrative totals fit below233472B/SM; actual static/reserved/granularity/device admission remains authoritative. Static shared size is not inferred from the witness. Actual host wrapper checks SM90/cooperative capability, successful dynamic shared attribute, actual selected kernel occupancy resident>=1, grid<=resident*SM, original grid/fixup/shape/dtype bounds and successful cooperative launch. `SSA_ADMISSION` prints actual dynamic/static/local bytes, registers, resident count, grid/SM/threads and selected B4/big/M after a successful query. A printed admission is not numerical qualification.

The unchanged final production prepare runs the actual new global factory/packers/scratch through the inherited Marlin-reference/private-guard/fresh/empty wrapper at128/32/33/256/1/8/16, observes every one of56 graph replays, checks output boundaries and guard tails, cleans private state and retains the295-second deadline. Native failure raises `SSA_STOP`; it cannot silently turn the candidate into a fallback measurement. Full operator tests and target device progress remain unrun here.

## Immediate handoff and stop rule

Captain can recheck authenticated quota/no-live and register this exact ordinary private layer candidate now. Return accepted own ID immediately, then own terminal association/raw log and all13tests/12benchmarks/12profiles, actual source/route/admission/startup and six `46*t_sliding+16*t_full` endpoints. First score is descriptive across jobs. A favorable substantial signal, preferably~3%+, earns a separately recorded matched confirmation decision; it is not causal proof or public authority.

Any hard host-build/admission/numerical/progress/setup failure stops this exact allocation without automatic rescue. Valid negative/subthreshold performance parks it without a geometry/spill sweep. No paired framework, proof of speed or further broad review is a prerequisite to the authorized first check. **Current measured savings: unavailable; no new job dispatched by this worker.** The commit containing this report is the implementation/evidence handoff; captain publishes its README link.
