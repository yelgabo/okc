# Original-K256 direct-global register weights: ready for ordinary private check

November 2, 2026. Allocation **7a1b048a**, [captain decision](../../../docs/steering/2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md); worker pickup23:09UTC. Existing OKC Developer owns implementation; **OK Driver alone owns judge dispatch, terminal archive and README/index**. Both earlier K128/shared-A implementations remain parked. No worker POST/poll, public/scored action, production-default edit or measured saving occurred.

The concrete ordinary layer candidate is [generated/candidate.py](../../../docs/steering/evidence/2026-11-02-worker-direct-weight-registers/generated/candidate.py), ready for the captain's one allocated private check. **Five focused CPU tests and eight actual selected secondary device compilations pass.** Target host compilation, cooperative admission, numerical/progress/startup/replay qualification and performance remain the ordinary check's gates. No proof of a win or general paired framework is a prerequisite.

## Frozen pins and exact treatment

| Artifact | SHA256 / bytes |
|---|---|
| Complete immutable control, public607eb52bdaf24160 | `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab` |
| Original selected primary native | `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71` |
| Direct-weight native | `f95fbb1124ac66b7dd2323a26a61368d0674e6ab079187c308f81ef77f2aba3a` |
| Full candidate | `a27a1baff54f5496e339044760d1a58927af8733fa6af6fadcc68c60dcf140bd` / **875672 LF bytes** |

Generator `tools/wgs_direct_weight_registers.py`, tests `tools/test_wgs_direct_weight_registers.py`. Original full prefix/notices are unchanged; a readable checked-anchor native transformation and factory override are appended. No compressed/opaque source, new packing, persistent BF16 expansion, output cache or general measurement framework. Base/native hash drift fails closed. The [archive manifest](../../../docs/steering/evidence/2026-11-02-worker-direct-weight-registers/archive-manifest.json) pins canonical source/diff, selected factory descriptors, exact secondary compiler inputs/logs, PTX/CUBIN and layout/codegen evidence.

Affected routes are B4/T32 and MAIN/T128,T256, both contexts. Keep **256 output rows, four compute WGs,64-token jobs,K256/16stages,NTT1..8, original prefix/range/split choices, RS-WGMMA math, packing, scratch, original finalization and one-CTA grid**. T1/T8/MID actual factory calls are byte-equal to control source/CPP/flags/exports/module names.

Only raw weight delivery changes: consumers read the same immutable prepared wAs/wBs vectors directly into the original register-prefetch/decode loop. The raw shared ring and32-thread weight-TMA producer are removed. **544threads=512compute+32X**; the sole helper is the original X producer. All three original X slots move together to the former ring base in producer, consumer descriptor and allocation. No geometry/cache/prefetch/unroll/register sweep.

Physical global address is `base+(((S.jE[u>>UBL]<<UBL)+(u&((1<<UBL)-1)))*16+physical_st)*WGS_STG+wg*WGS_PIECE`, with original UBL5A/4B. A segment computes its immutable expert/unit base once and passes actual `s0` or loop `st` into the stage lambda. **`seq` selects X-slot/parity only; `seq%16` is never a weight stage.** Reordered/split ranges make that substitution invalid.

First code and raw/SC6 scale loads precede the X wait. The existing next-load-group register prefetch remains. Attributed emitted PTX shows `ld.global.v4.u32` code loads and the corresponding global scale loads before the X `try_wait`; it does not establish useful dynamic latency overlap. Remaining bulk TMA code is solely the X producer. Selected PTX contains one X bulk-copy instruction site per bool entry; unrelated FAST1/fat implementations remain unchanged and inactive under actual WGS/FAST1=0 flags.

## Mathematics, ordering and lifetime

The full code/scale decode loop, `dq4`, `bmul2l/h`, raw and SC6 offsets, group16 pairing, register-A fencing, RS-WGMMA accumulator operations, global postscale, BF16 boundaries, SiLU and final normalization/combine remain the original code. Tests compare complete numerical and finalizer regions, not only helper names. Original packing and scratch functions are unchanged rather than adapted from a parked K128 implementation.

Prepared weights remain immutable and retained in the returned state's `ma_w` for eager and captured reads. The global code/scale reads require no activation arrival, so they may issue before X-full; no X operand is read by WGMMA before observing X-full. Startup routing/gather publication and the544-thread full join preserve initialized table/barrier visibility.

Only the now-dead raw-shared full/empty waits, producer and raw-release proxy fence/arrival are removed. X full/empty barriers, writing-side global proxy fences, inter-CTA release/acquire, async register fencing and terminal waits remain. Every consumer WG executes `wg_wait<0>()` before its threads release an X slot; X-empty requires all512 consumer arrivals. The [PTX async-proxy rule](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-async-proxy) remains relevant to retained X publication/retirement. Test passing does not establish ordering by itself.

Original phaseB waits for8 publications at `doneAs[j*16+st]`; phaseA publishes at **`ub>>1`**. All16-stage completion/reset/epoch/split/fixup logic, reachable FCI4 deferred flush and phase-end drain remain byte-identical. No SS/K128 counters or stage ordering are imported. Empty/tail work keeps the original paths. A lost weight producer cannot strand an arrival: full join is reduced to544, only the X helper remains, and raw barriers are neither initialized as active protocol participants nor awaited.

## Focused CPU and compiler evidence

**5/5 CPU tests pass**; [output](../../../docs/steering/evidence/2026-11-02-worker-direct-weight-registers/cpu-tests.txt). Actual generated loader/selector is exercised for raw and SC6 at1/8/16/32/128/256/384. Tests compare unchanged short factories, packing/scratch/startup functions and original mathematical/range/finalizer bodies; enumerate global offsets for every local expert/unit/stage/WG in both contractions and formats; reject `seq%16` on skewed/reordered/split examples; cover every per-lane code/raw/SC6 packet byte and alignment; verify sole-X active delivery, X retirement, counter and capacity invariants. CPU/source evidence does not prove GPU numerics, scheduling or speed.

Actual B4/MAIN false/true entries compile in both raw/SC6 factory branches: **eight entries**. Installed NVRTC13/PTXAS13, SM90a, disclosed virtual headers/C++20/declaration-only adapter; no local target NVCC/host build claim. Effective factories use host/CUDA C++20, `-O3`, SM90a codegen, lineinfo and original selected defines. No blanket fast-math addition. All entries emit maxntid544/minnctapersm1 and zero C7513 warnings.

| Selected family/format, both bool entries | Registers | Barriers | Stack B | Spill stores B | Spill loads B |
|---|---:|---:|---:|---:|---:|
| B4 raw/SC6 |96|16|16|60|84|
| MAIN SC6 |96|16|40|224|616|
| MAIN raw |96|16|32|196|384|

Previously archived same-C++20 c7 device controls report B4 96regs/16barriers/16stack/120store/216load and MAIN96/16/8/20/52, [reference evidence](../../../docs/steering/evidence/2026-11-02-worker-wgs-k128-full-reuse/reference/compile-result.json). They were not rerun here and are not fresh target host measurements. MAIN spilling grows substantially; lower B4 spill counts are not a speedup. No tuning follows these static counts before the funded check.

Emitted witness confirms original `sizeof(Smem)=22240`, aligned header22272, JMAX88, original fixup5246336floats,544threads, **three32768-byte X slots=98304bytes**, hence **120576dynamic bytes** in both formats. No new fields. With illustrative1024B reserved/block,121600B/CTA means two would exceed233472B/SM; no doubled occupancy claim. Actual selected-kernel static/reserved/granularity and resident>=1 query remain authoritative.

## Hypothesis, target gate and stop

Logical HBM payload does **not** decrease. Direct global latency, request/scoreboard/cache pressure replace LDS plus bulk TMA, and the independent weight producer's three-stage lookahead is lost. This hypothesis asks whether removing raw shared delivery/waits helps enough despite those costs. The SS loss does not identify a cause or predict this result. The captain's~11.37% fully exposed affected-activity conversion for3% layer gain is a conditional required-effect scenario, not a forecast or removable-time bound.

Ordinary inherited final `prepare` remains unchanged: actual selected factories, independent Marlin reference/private scratch guards/fresh inputs/empty transitions, all56 observed graph replays and295-second deadline. Native failure raises `DW_STOP` rather than silently benchmarking fallback. Host admission retains SM90/cooperative/shape/dtype/grid bounds, actual dynamic-memory attribute, resident>=1/grid<=resident*SM and successful cooperative launch. `DW_ADMISSION` identifies actual selected B4/big/M, SM/grid/544threads, dynamic/static/local bytes, registers and resident count after a successful query.

Captain can verify exact source and fresh authenticated quota/no-live, then register the one allocated ordinary private layer file. Capture accepted own ID, owning raw terminal/association/log and13passing tests/12passing benchmarks/12complete profiles, actual native selection/admission/startup and six46:16 endpoints. First cross-job score is descriptive. A substantial favorable result, preferably~3%+, earns a separate matched-confirmation decision, not public authority.

First target setup/math/progress/admission failure closes this fixed allocation without rescue; valid weak/negative parks it without tuning. No new observer, proof of a win or source-research loop is required before measurement. **Current measured savings: unavailable.** This report's implementation commit pins the handoff; README/index remains captain-owned. Prior losses and unrelated WIP are preserved.
