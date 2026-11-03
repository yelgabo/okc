# Fixed native K128 / two-front implementation: ready for first private measurement

November 2, 2026. Implementation owner: existing OKC Developer. Allocation: [captain takeover](../../archive/docs/steering/2026-11-02-driver-captain-takeover.md), initial commit **88fffb5f**. Captain **OK Driver** owns all judge dispatch and README/index publication. No worker judge action, public submission, production-default edit or measured saving is claimed here.

The concrete full-layer candidate is [generated/candidate.py](evidence/2026-11-02-worker-wgs-k128-full-reuse/generated/candidate.py). It is ready for the captain's first authorized ordinary private layer check. This completes implementation/source/CPU/secondary-device-compile work; target host compilation, real occupancy, numerical/replay qualification and performance remain gates executed by that check. There is no preliminary MoE run or new measurement framework.

## Exact frozen source

| Artifact | SHA256 / bytes |
|---|---|
| Immutable full control, public job `607eb52bdaf24160` | `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab` |
| Actual original primary native `MOE_CUDA_SRC` | `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71` |
| New selected native | `6366934e170d7abf386869b4c22874f373d8b1d812ccff3400f7753a665c6a4a` / 155456 LF bytes |
| Full candidate | `440f41b118eeb3676867e15d109c2d5f7da0213873f84a17c9d68d1af868703f` / **885489 LF bytes** |

The generator `tools/wgs_k128_full_reuse.py` keeps the entire original prefix and notices byte-identical after LF normalization and appends readable checked-anchor transformation/packing/factory code. No compressed or opaque embedded source. It rejects an unexpected base/native pin. Canonical source, native diff, selected factory flags, exact compile records and compressed lossless PTX/CUBIN evidence are in [the evidence directory](evidence/2026-11-02-worker-wgs-k128-full-reuse/archive-manifest.json). Only canonical `generated/candidate.py` is the handoff payload; earlier scratch-generated copies are superseded.

## Fixed treatment and expected costs

Only WGS B4 at T32 and MAIN at T128/T256 change, both sliding/full contexts. T1/T8/MID selected native source, CPP source and flags compare exactly equal to the control through the actual `_fork_native_moe` entry. Unrelated layer operators remain unchanged. The original c7 primary native is transformed; the later MID helper literal is not substituted.

The fixed geometry is two compute warpgroups, 128 output rows/CTA, Kstage128, 64-token jobs and count-dependent NTT1..8. The CTA contains 256 compute plus 64 producer threads; grid is two CTAs per SM, at most 264, with 265 fixup slots. Separate header arrays remain separate: no weight/partial alias. No tile, cap, mask, stage or compiler sweep.

Hypothesis: independent producer/dequant/compute fronts hide more stalls while preserving 64-token weight reuse. It does **not** double total compute warpgroups/SM. Against the original one-CTA/four-WG arrangement, equivalent 256-output/K4096/64-token work has unchanged logical weight bytes, twice the X-copy requests (512 KiB to 1 MiB), four times the producer stages/handshakes, a doubled grid and different split/finalization costs. Cache behavior and exposed latency remain unmeasured. These costs are included in the ordinary full-layer result.

The captain's conditional three-load model gives 2.62%/5.47%/8.57% layer improvement if 10%/20%/30% expert activity reduction converts fully to endpoint saving. Approximately 23.75% is needed to bridge the sampled cutoff under that model; these are required-effect scenarios, **not predictions or measured removable-time bounds**. MID is unchanged; four-long-load opportunity arithmetic does not apply.

## Numerical and lifetime obligations implemented

- K128 packing is a new lossless permutation, rather than cutting K256 packets in half. Each WG owns 4096 code bytes plus 512 raw scale bytes, or 384 SC6 bytes. Four load groups each consume two K16 steps; every original group16 scale stays with its codes. SC6 reconstructs original raw E4M3 bytes before the existing decode and BF16 arithmetic. Original FP4 values, global scales, BF16 boundaries, SiLU/up ordering and output normalization/combine are retained.
- The activation descriptor pitch is 2048, and source gather addresses, down-intermediate store, row permutation and packed operand mapping all change consistently. Actual generated gather expression is checked against an independent logical-K reference; the old gather expression and old 4096 descriptor fail discriminating negatives.
- There are 32 K stages/unit, 64 A units/job and 32 B units/job. A B-stage waits for four releases (two contributing A units times two WGs); `ub >> 1` maps those publications to 32 counters/job. Initialization, generation parity, resets and scratch capacity use the new dimensions.
- Work ranges use 64-bit products before division. Split-unit ownership, last-stage selection, pending counts, unit completion and fixup addressing use 32-stage units. The designated last stage is 31; completion target is 32. No inherited 16-stage completion or G224 eight-release target remains.
- K128 has commit groups ci0..3 with GS2. The pending previous segment is flushed at ci3 after the existing wait1, before replacement. The primary native's full `wg_wait<0>()` before releasing `BX_E` is retained, so asynchronous consumers retire before X-ring reuse. Writing-side `fence.proxy.async`, weight/full-empty handoffs, release/acquire intermediate publication, cooperative barriers and terminal drain remain. Test agreement alone is not offered as an ordering proof.
- B4/MAIN finalization maps the original logical 16 warps onto two slices of the actual eight warps; logical reductions retain the original order. GPU numerical checks remain required.

## CPU and compiler evidence

**7/7 CPU tests pass** in `tools/test_wgs_k128_full_reuse.py`; [exact output](evidence/2026-11-02-worker-wgs-k128-full-reuse/cpu-tests.txt). Tests execute real CPU Torch packing for raw and SC6 scales, decode produced bit positions against independent original logical codes/scales, verify gate/up row semantics and K128 gather/descriptor mapping, exercise actual selected factory calls, test scratch dimensions/independent allocations, odd counts33/257 and legal 64-token job tails, stage/split/fixup/publication capacities, plus discriminating old-address/descriptor/stage negatives. CPU evidence does not establish GPU scheduling or numerical equivalence.

The actual B4/MAIN false/true selected entries compile with installed **NVRTC13 + PTXAS13**, SM90a, disclosed C++20, virtual headers and the existing declaration-only lookup adapter. Both produce `.maxntid 320` / `.minnctapersm 2`, with **zero C7513 serialization warnings**. Actual target factory uses `-O3`, C++20 in both host and CUDA flags, source/flags keyed module names and the original selected defines except NW8. There is no blanket fast-math addition. Local evidence is secondary device-only compilation; target NVCC/host compilation has not run here.

| Selected family, both bool entries | Registers | Barriers | Stack B | Spill stores B | Spill loads B |
|---|---:|---:|---:|---:|---:|
| Fresh original B4, same secondary C++20 | 96 | 16 | 16 | 120 | 216 |
| Fixed K128 B4 | 96 | 16 | 24 | 108 | 200 |
| Fresh original MAIN, same secondary C++20 | 96 | 16 | 8 | 20 | 52 |
| Fixed K128 MAIN | 96 | 16 | 24 | 68 | 172 |

The original B4 is **not** the MAIN 8/20/52 baseline. MAIN spill cost increases materially; no claim it is hidden. Captain explicitly permits this admissible fixed candidate's first measurement rather than a spill-tuning loop. The fresh reference uses C++20 in the secondary driver for comparison; original factory flags are recorded unchanged in the reference evidence, so this is not claimed to reproduce a fresh target control host build.

An emitted constant-only layout witness (not a changed kernel body) gives `sizeof(Smem)=21216`, aligned header **21248**, rings **76800**, JMAX88, fixup4345636 floats. Dynamic shared memory is **98048 B/CTA**. An illustrative 1024-byte reserved allocation gives 198144 B for two CTAs versus 233472 B/SM; real device granularity/static/resource admission remains authoritative. Constant witness bytes and original layouts are archived per family.

## Reused real startup and target decision

The unchanged final production `prepare` invokes `_release_expert_self_test` with the current global factory, packing and scratch implementations. It creates guarded private mutable scratch with no production-storage alias, compares to separately prepared Marlin reference, starts from empty routes/fresh scratch, tests native-selected sizes `(128,32,33,256,1,8,16)`, uses legal distinct routes and fresh BF16 inputs/FP32 route weights, captures actual treatment calls, copies fresh values at fixed capture pointers, observes every output in **56 graph replays**, checks repeated empty/helper outputs and guard tails, then cleans private state. The 295-second deadline remains. Native failure raises `K128_STOP` rather than silently benchmarking fallback.

Actual host launch checks SM90/cooperative capability, grid/fixup/shape/dtype bounds, successful dynamic-memory attribute, **`cudaOccupancyMaxActiveBlocksPerMultiprocessor` resident >=2**, grid <= resident*SM and successful cooperative launch. One-time `K128_ADMISSION` lines identify B4/big/M, SM/grid/threads, dynamic/static/local bytes, registers and actual resident count before the first launch of each bool entry. These lines report a passed admission query, not numerical success. Startup success must additionally print `G_RELEASE_STARTUP_OK native ... 56 verified_replays`.

Captain next action: recheck authenticated quota/no-live, register this exact ordinary **private** layer file, report accepted own ID, then collect owning terminal raw receipt/log and all13tests/12benchmarks/12profiles. Evaluate actual selected treatment and complete per-load `46*t_sliding+16*t_full` costs. Report absolute microseconds, score/CI and aggregate log contribution, including regressions. A first cross-job official score is descriptive; if substantially promising, captain may use the second allocated check for matched complete-layer confirmation with independent controls and duplicate-noise/regression guards. No extra source-review approval cycle is requested here.

First invalid host-build/admission/numerical/setup result returns an exact blocker for captain decision; no blind retry. A valid subthreshold/regression result parks this fixed implementation without parameter fishing. No public/scored action or production promotion follows automatically. **Current measured savings: unavailable; no target job/result owned by this implementation report at publication.**

The implementation commit containing this report is the source/evidence handoff pin; captain indexes it separately because README ownership remains captain-held. All earlier losing versions and unrelated work are preserved.
