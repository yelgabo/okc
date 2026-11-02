# T1-only three-launch expert prototype: offline implementation

November 2, 2026. Allocation **60adf8cdb3f9060733085c4a5d8406f100d01a39**, captain PR-terminal/next-implementation report, using researcher screen6b14088. Work began **15:14:42UTC** after terminal335887c was complete. The PR allocation remains CLOSED; none of its source or artifacts is repaired or retried here.

## Decision and exact source

**OFFLINE_PROTOTYPE_IMPLEMENTED_CPU_AND_DEVICE_COMPILE_PASS; NOT_TARGET_QUALIFIED.** This is ONE fixed T1-only implementation, not a performance win or target allocation. No GPU execution, Driver query, challenge CLI, installs, host syntax/build/link, scored/public/default changes, subagents or new threads. No automatic target request or follow-on test.

- Base exact ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab,869076LF bytes, muse_decode_layer/cand/g1b4_stage_release.py.
- Full offline snapshot generated/candidate.py: **faed74a8f6f10c87927555267ad3b7e2804db84b73e3dd12677968c7b646e266**, **884252LF bytes**. Exact original prefix, provenance and full licenses retained; no compression/notice stripping or production-file edit. Suffix removal exactly restores ba9.
- New native selected.cu: **dbdd3381cc9b823afb9e7d8678bba536cd5d6e3731896153aee9908652237447**,12108B. CPP binding declaration64986286b26616a9fe70c885b53aee9c409af0ed694cd16f1ece47df62193b75. Source/diff/append/generation pins are under evidence/2026-11-02-worker-t1-three-launch/.
- Owned implementation: tools/g_t1_three_launch.py, tools/t1_three_launch.cu and tools/test_g_t1_three_launch.py. Five decoder/MMA/rounding helpers are copied byte-exact from the actual selected helper source; their individual hashes are retained in generation.json. The selected SCDK scale embedding and SILU_DIV intrinsic remain explicit.

Full-snapshot git whitespace checking reports inherited '# ' license-comment lines in the exact base prefix. Those bytes are deliberately retained, not stripped to satisfy formatting. The new handwritten tool/template/test/report paths are checked separately; no unrelated formatting repair is made.

## Actual callable and state isolation

The appended prepare calls the actual final original prepare, including original normal preparation and startup, BEFORE activation. It enables the new state-local adapter only for native-eligible states with the exact selected PRMTSC1 suffix and existing packed-weight vector. Other scale layouts/native fallback keep the original selector. No state-global setter, numerical precomputation, external allocation, graph ownership, route/output cache or capture query is introduced.

Only _moe(M=1,state) selects the adapter. T8 and every other M delegate to the captured final original _moe, not a guessed alias. The adapter retains the original okc_run signature and delegates unsupported dimensions, precision, contiguity or local-range ABI unchanged. Existing immutable unit-packed weights w[10]/w[11], FP32 global scales and BF16 post-expert norm are passed directly; no new packing or scale conversion. The selected MOE_HAS_PDL=False is asserted in the actual-layer CPU test; no new setter is required or invoked.

Each prepared adapter owns ordinary torch.empty scratch: BF16 inter[8,4096]=65536B; FP32 y[8,4096]=131072B; FP32 partials[8,256]=8192B; total **204800B/200KiB**. Independent prepares allocate independently; adapter deepcopy shares immutable modules and copies ordinary scratch with the memo. Original ma_scratch stays available for unchanged fallback/other loads. These are ordinary framework-owned tensors, not a replacement memory ownership system.

## One fixed three-launch implementation

All three launches use the caller's same current stream with ordinary launch attributes. The complete predecessor kernel, not a cross-grid flag or spin loop, publishes each phase. New intra-CTA reduction storage uses full __syncthreads; no asynchronous producer accesses this new shared region. All original seven factory bodies/fences/flags are unchanged, so no incumbent fence was dropped.

1. **Gate/up:** grid(512,8),256threads. Each fixed slot lane skips nonlocal IDs using the explicit0<=id<32 check, never modulo routing. Eight compute warps cover the selected full K4096 decomposition for one8-coordinate intermediate tile. The selected four-accumulator MMA grouping and warp-order FP32 sums are retained. Scaled gate and up each round/widen BF16 before scalar __expf/__fdividef SiLU; activation times up is BF16-stored.
2. **Down:** grid(256,8),256threads. Same packed decoder/MMA structure produces16 output coordinates per tile. Scaled down accumulators round/widen BF16, multiply BF16-rounded(1+BF16normw) in FP32, and store FP32 y. Each tile stores a complete16-coordinate FP32 squared-norm partial with the selected pair-square/XOR reduction grouping. No atomics or cross-CTA waits.
3. **Complete per-expert norm and combine:** one256-thread CTA. Each thread loads one of256partials for each valid slot. Warp reduction plus ordered warp sums yields that expert's complete sum; rsqrt(sum/4096+1e-8) is published in shared memory. Only then accumulate FP32 weighted y in original slot order0..7 and store final BF16. No global norm substitution or hidden fourth kernel.

Invalid slot scratch is never consumed. Valid slot tiles overwrite every required inter/y/partial coordinate on every call. All-empty routes still execute the three ordinary launches; the last writes all4096 positive zeros without reading stale/unwritten invalid-slot data. T1 duplicate local IDs remain slot-distinct; no compaction/preprocessing kernel is hidden.

## Executed CPU and source contracts

**11 tests PASS in15.706s**, final cpu-final.log. These execute the real generated prepare and _moe/adapter and the actual selected layer expert-dispatch AST block, with mocked CUDA/native boundaries. The base final prepare, Lt-plan integration and original startup invocation are preserved; this is not CPU execution of CUDA preparation.

Coverage includes two independent preparations/interleaved calls, private scratch and deepcopy, original prepare failure, real T1 versus T8 layer expert block, all other selected sizes and unsupported fallback, complete output/weight/input/scale ranges and zero token padding, all16FP4 code/sign values through actual _stored_pos DQ7 bit mapping,119 selected normal scale-byte embeddings, BF16 ties/signed-zero and gate/up/down/gamma boundaries, complete per-expert norm versus forbidden norm-after-combine, slot-order counterexample, and fresh/stale scratch models with every local-pair count0..8. CPU source negatives reject lost route guard, modulo mapping, missing rounding/norm, changed launch order/count and accidental T8 adapter selection. A deliberately widened T8 selector is also executed and distinguished from the actual selector.

The original seven capture dictionaries are byte/flag/function-equal before/after in original-seven-factories.json: five selected families plus two retained historical short-RMW capture requests, not seven new candidates. The eight existing M-to-native-family requests remain unchanged. New T1 adapter selection is separately exercised at the actual callable, rather than assuming unchanged factory capture proves the treatment route.

Initial CPU fixture failure was a mocked Lt object's ctypes-pointer deepcopy; the fixture now installs the actual original _GraftLt.__deepcopy__, not a treatment change. A new semantic-source negative then exposed an omitted route-guard anchor in the offline validator; the guard was added to that validator. Initial logs are retained. The CUDA implementation, geometry and source hashes did not change during these test corrections. Host arithmetic models use explicit FP32 rounding and host exp/sqrt; they do NOT establish CUDA intrinsic accuracy, contraction behavior, MMA equivalence, graph replay results or numerical qualification.

## Fixed secondary compile and actual resources

Successful single source compilation **15:22:48.631–15:22:49.121UTC**, well before the30-active-minute aim. The first invocation at15:21:34 omitted the established LD_LIBRARY_PATH and NVRTC could not open its already-installed builtins. That loader failure is retained under initial-builtins-failure/ and first-compile.log. Identical source was invoked with the existing recipe's LD_LIBRARY_PATH=/root/.cache/okc-cuda130-88/nvidia/cu13/lib; no package/compiler/geometry change or native syntax correction. No compiler sweep.

Recipe: exact installed adapted-driver.py SHAe695ef03bda22d86167b829618429c7d139c23ad9cf96294edd1211388d5620c, NVRTC13/PTXAS13, sm90a, original virtual headers and actual ordered selected T1 defines. Only the name-expression list is adapted to the three new entry points. No declaration/layout adapter is needed for this new prefix. Compile inputs, adapted driver, NVRTC/PTXAS logs, complete PTX and cubin and manifests are archived; host code after torch/extension.h is explicitly NOT compiled.

- gate_up:64registers,1barrier,4096Bstatic shared;0stack/0spill store/0spill load.
- down:64registers,1barrier,4096Bstatic shared;0stack/0spill store/0spill load.
- norm_combine:32registers,1barrier,64Bstatic shared;0stack/0spill store/0spill load.

These are observations, not a universal zero-spill/occupancy gate or speed proof. All three PTX entries have maxntid256. Static PTX inventory retains32 MMA and128 BF16x2 multiplies in each matmul entry, gate scalar exp/divide and required BF16 conversions, and complete per-slot rsqrt in finish; no local loads/stores or atomics. Counts are NOT cycles or a timing forecast. Full PTX199676B SHA b1c93ac1656ad766c8575dd31f154ea345fedd2dedbb27da41db19f158b43055; cubin157504B SHA80231b74a2fedfd6ded096f46f07ba7040553d1ef3954092238cbba64ad62ccd. No SASS equivalence or target occupancy claim.

Reproduce only this offline compile with: env LD_LIBRARY_PATH=/root/.cache/okc-cuda130-88/nvidia/cu13/lib python3 tools/g_t1_three_launch.py --compile --out <owned-evidence-dir>. Do not interpret this as target dispatch permission.

## Real competing costs, gaps and stopping boundary

This changes scheduling, not the fact that tiny already owned output tiles. It incurs **two extra launches**, loses per-expert A/B overlap by completing all A before B, and serializes complete normalization/final consumption through one CTA. The fixed grids dispatch4096A+2048B CTAs even when many slots immediately return. Weight units now use ordinary global fragment loads instead of the persistent TMA/helper ring; CTA-private input fragments reload per tile rather than persisting across that CTA's units. Cache/traffic/launch/normalization exposure is unmeasured. Existing inter/y materialization cannot all be credited as new or saved traffic; additional norm partials and retained original scratch also cost space/traffic. No static count or activity attribution establishes a net gain.

Only T1/D1 is affected; T8/V1, all larger-token schedules, front/router/unpool and all other layer computation are unchanged. No fresh route populations, extra-launch latency, whole-call46:16 score, >=1% opportunity or top100 gain has been measured or forecast. The researcher's old route examples are not substituted for fresh layer routing.

New host binding compilation/link/load and target Torch/NVCC ABI, real CUDA/replay outputs, new norm-tree numerical tolerance, real memory bounds/retirement, prepare budget and actual route/weight traffic remain untested. New host extra_cflags explicitly select C++20 rather than repeating the closed PR standard mismatch; that is an inventory item, not host-build qualification. Original startup runs BEFORE adapter activation and therefore does not certify the new kernels.

Return ownership to captain for independent source/priority review only. Any concrete packing/math/order/compile contradiction stops this exact prototype. Further build/runtime/measurement work requires a separate explicit allocation; no automatic readiness cycle, tuning variant, T8 expansion or target-check request. README/shared run/ruled-out ledgers and concurrent WIP are untouched. Publication manifest and enclosing commit pin the final handoff; prior PR terminal/retry artifacts remain immutable.
