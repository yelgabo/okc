# Lossless factorized expert-output storage

November2,2026 Pacific. MOE-FACTOR-Y-01 is the next owned B4/MAIN implementation. Public5df257af remains5788.720720us; user target5559.8us remains open. The prior pair lookup and direct input screens passed correctness and lost their complete matched gates. Their exact variants remain stopped. General owns the separate T8/T16 row-owned build. No GPU job was active at the design freeze.

## Exact intermediate instead of a wider product

The selected C7 phase-B epilogue computes d=BF16(FP32accumulator*global_scale), then y=FP32(d*gamma), with gamma=BF16(1+raw_norm_w). It stores y in a FP32 route buffer. MAIN reads y twice for full-width RMS and route combination; B4 stores fixed sum-of-squares partials and reads y during its final pass.

The candidate stores d in BF16 and reconstructs the same FP32 product y when reading the buffer. Every finite BF16 d round-trips exactly. The reader recomputes the same immutable gamma and uses an explicit FP32 round-to-nearest multiply, retaining the old store's rounding boundary before norm or combination. No gamma folds into W2, no dot-product order changes, and there is no BF16 rounding of the post-gamma product. Original OKC_YBF=0 remains unchanged; a separate factorized-storage branch names the new representation.

This halves the y-buffer value payload in stores and reads. MAIN's route-buffer payload changes from12*R*4096bytes to6*R*4096; B4's one-read payload changes from8*R*4096 to4*R*4096. R is the actual local route count. B4 norm partials, fixups, routing and all other traffic remain. These are byte counts, not HBM traffic or an exposed-time estimate. MAIN performs the gamma product twice rather than once, and its reader gains gamma loads/conversions. B4 keeps its original exact norm partial calculations. Added reader instructions can outweigh the smaller buffer.

The previous pre-gamma treatment changed W2 arithmetic and failed. This candidate changes storage after the existing required BF16 down boundary while reconstructing the identical FP32 product. The prior direct-input alternative moved repeated scattered loads onto the producer and lost. This candidate retains input gathering and changes neither producer scheduling nor ring geometry. It is a different coherent work/representation test, not a mask or rescue.

## Lifetime, fallback and correctness

Use the existing y allocation, writing compact BF16 rows only in the existing route-output region. Keep all fixup/count and B4 partial offsets at their original FP32 addresses beyond this region. No new allocation, alias with another owner, preparation cache or launch is needed. Every valid output row is fully overwritten before the unchanged fences and completion publication. The finalizer reads only valid routes after the same barriers. Mixed-size replay can alternate original helper FP32 storage and candidate WGS compact storage because each call overwrites every valid row and completes before scratch reuse. Empty-local routing still produces zero output.

Keep packed weights, preparation flags, local expert IDs, routing/slot order, phase-A intermediate, BF16 down/gamma boundaries, RMS tree/epsilon, route combination, producer/consumer roles, proxy fences, readiness/reset/generation, grid residency and graph lifetime. Preserve the original helper source at T1/T8/T16. Audit every selected WGS read/write of y; dormant non-WGS/fallback bodies retain their original representation. Passing numerical tests alone cannot justify a missing read/write or changed mathematical boundary.

Before GPU dispatch, prove the BF16 round-trip and reconstructed-product equality, including signs, zero, subnormal and overflow behavior; show an injected post-gamma BF16 round differs. Compile/inspect both B4/MAIN bool entries, verify selected source/flags and retain diagnostics. CPU execution is separate from H200 evidence. Independent review remains unavailable under the no-new-agents instruction.

## One finite complete screen

Original three banks, normal/skewed routing, all54scored/neighbor cases, independent duplicate controls/candidates,378fresh observations,216captures,1152cold timings,54restores and scratch/input/graph checks. Keep the0.015global/0.075row bounds, exact source/record/coverage gates, every-bank/allfour-cross complete-six gain at least1% and above twice duplicate-log noise, no affected1%regression and no unchanged-path drift. Preparation stays295s, with bounded partial evidence retaining failure status. Success requires held-out confirmation before integration or another public request.

Stop this exact factorized-storage implementation on source-policy, mathematical/layout/progress/lifetime, source attribution, numerical, budget, coverage, drift, regression or benefit failure. No post-gamma rounding fallback, packing/geometry/cache sweep, larger guard, subgroup mask or blind retry. Local spill observations do not decide target speed. Publish exact source, verify remote bytes, perform fresh shared job/quota preflight and persist dispatch/accepted own ID before one private check.

The [PTX8.7 FP32 multiply contract](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#floating-point-instructions-mul) defines the explicit rounding and default subnormal behavior. Reconstructing identical operands and using that multiply preserves the original stored product boundary.

## Frozen source and local qualification

Private source78506b743ace84e1fc0b088fe7fe3b23009f3de0bf5136e20413504026e07492,348783bytes. Runtime678e4f28295fe563de898d93a496b8bc0fa07b218894520047eb9697a5b80e51. Nativeecd1df116a7dd43d8a76125edafd385c8e9380d3d525ed2dc4ce6d0973d13155 versus originalc7c201ae. [Full source/flags/routes](evidence/2026-11-02-moe-factor-y/paired.manifest.json). Original packed weights, preparation and helperb8ed8543are unchanged.

Eight CPU unit tests pass:2finite-BF16/product/source,4actual Torch orchestration/ownership/failure and2strict full/partial readers. The separate actual-native CPU helper witness passes2088960product comparisons and8192paired row/column loads; an injected post-gamma BF16 round returns1 as expected. This is a finite operand/weight selection plus an algebraic identical-operands proof, not exhaustive all-pairs testing. CUDA arithmetic and synchronization are absent from the CPU facade. [Actual native CPU witness](evidence/2026-11-02-moe-factor-y/native-host/summary.json).

Four local CUDA12.8.1 SM90a entries compile with exit0. Both candidate B4/MAIN bool entries use96registers/16barriers andzero stack/spills. Control B4 retains8stack bytes/4spill stores/32spill loads; control MAIN haszero. All entries retain512static HGMMA and48full drains; emitted operations and complete logs are retained. These are local observations, not H200 parity or a timing win. [Resources](evidence/2026-11-02-moe-factor-y/resources.json), [complete archive](evidence/2026-11-02-moe-factor-y/archive-manifest.json).

The runtime keeps seven actual compiler-request pins, four independent owners, original256word lookup buffers,54case/36row coverage and the unchanged complete-six/drift gate. All preparation/cleanup/serialization/budget failures retain bounded partial evidence and remain failed. The full-reader fixture uses retained target timing rows under an explicitly synthetic source association to test report size and rejection. No new GPU score follows from these tests. Author checked the proof and sources; independent review remains unavailable.

The local CUDA12.8.1 header confirms the raw-store conversion uses cvt.rn.bf16.f32 on SM80+, matching the incumbent fork_bf16 primitive. [Conversion check](evidence/2026-11-02-moe-factor-y/conversion-contract.json). No conversion flag or tolerance changed.

## Accepted private screen

Own7479372115c64702 accepted for frozen78506b74/348783bytes, publishede2b3c17f. Fresh shared list hadzero live jobs, with10MoE private checks left before dispatch. Accepted own source/job association is archived. Follow only this ID with one foreground waiter. No target result yet.
