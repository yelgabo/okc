# Application ledger

November 2, 2026. Audited source and evidence hashes are in [catalog.json](catalog.json). Every recommendation is a conditional optimization choice. "Present" describes inspected source; it does not certify optimal compiled behavior. "Parked" describes the exact earlier implementation and its recorded scope.

## Decisions against our code

| ID | Research recommendation | Application decision |
|---|---|---|
| R01 | Establish the baseline and hotspot | Reuse SGLang and incumbent receipts; verify owning IDs and actual selected route. See [profiling](profiling.md). |
| R02 | Treat H200 as Hopper with more bandwidth | Apply SM90a and memory sizing; actual device/traffic determines relevance. See [hardware](hardware-and-numerics.md). |
| R03 | Use native BF16/FP16 and FP8 Tensor Cores | BF16 MoE contract stays. FP8 exists in attention; replacing MoE activation precision is incompatible. |
| R04 | Treat NVFP4 as W4A16 | Present. Packed codes/scales become BF16 operands during compute. No native Hopper FP4 path. |
| R05 | Fuse packed loads, scales and dequant | Present. WGS uses register-A decode; the research's decoded shared bank is optional. |
| R06 | Compare predecoded, separate and fused paths | Laboratory diagnostics only. Persistent expanded BF16 weights violate the task; a public candidate must retain eligible storage. |
| R07 | Confirm WGMMA/TMA in SASS | Apply per selected entry and compiled flags. Existing source syntax alone is insufficient. See [validation](validation.md). |
| R08 | Prefer WGMMA over older MMA | Already used in WGS. Preserve the separately selected tiny/MID path until a complete replacement wins. |
| R09 | Build a producer/consumer asynchronous pipeline | Present. Another pipeline must name its improvement over the existing raw ring and RS-WGMMA loop. |
| R10 | Sweep stages and prefetch depth | Conditional on exposed feed bubbles. Existing ring depth depends on configuration. Avoid reopening parked sweeps from a generic range. |
| R11 | Sweep CTA M/N/K and accumulators | Require selected-route utilization and net budget. The exact K128 full-reuse job `bf8671b4bfa845a2` is parked. |
| R12 | Inspect shared banks and use swizzle/padding | Apply to actual access patterns. Keep operand descriptors valid; no padding change was justified here. |
| R13 | Reallocate producer/consumer registers | Candidate when the selected resource/stall evidence supports it. `setmaxnreg` availability does not prove dynamic allocation works as intended. |
| R14 | Maximize useful compute rather than occupancy | Apply. Record actual dynamic shared memory, register demand and resident-grid progress. |
| R15 | Cluster/multicast repeated expert weights | Conditional. Prior skinny coverage screen stopped before implementation. Require repeated simultaneous consumers and cluster occupancy evidence. |
| R16 | Expert-major CTA ordering and L2 reuse | Expert jobs already exist. Reorder only with actual DRAM/L2 and tail evidence; logical repeated requests are insufficient. |
| R17 | Persistent grouped scheduling | Present in native. A new complete grouped engine must beat it while paying both GEMMs and Muse glue. |
| R18 | Global work queue or local range plus stealing | Conditional on a demonstrated locality/imbalance tradeoff. Price work acquisition and reset. |
| R19 | Expert-aware Split-K | Conditional on insufficient output parallelism. Charge reduction, precision and workspace costs; preserve distinct per-load decisions. |
| R20 | Stream-K tail balancing | Tested implementations are parked; retain the specific fixup/setup reopening triggers in [scheduling](scheduling.md). |
| R21 | Fuse bias, activation and epilogues | SwiGLU and norm/combine integration already exist. Preserve BF16 boundaries and expert-wide RMS before mixing. |
| R22 | CUDA graphs | Already the judge's timing contract. Adding capture is not a new speedup. Validate fresh replay outputs and buffer ownership. |
| R23 | Add streams/events for independent work | The layer already overlaps router work. New overlap needs explicit independence and lifetime/progress proof. |
| R24 | Systems timeline and NVTX | Useful missing target evidence. Templates are documented; no Systems capture ran in this intake. |
| R25 | Compute roofline and stall counters | Useful missing target evidence. Inspect issue-active rate before attributing stall percentages. No new counter data is claimed. |
| R26 | Use production and synthetic routing distributions | Existing harnesses cover routes; preserve real counts and legal top-k/EP8 semantics. Toy histograms cannot choose production dispatch. |
| R27 | Report shape/config/resources/median/p95/error | Apply to future experiments with source identity and fresh replay checks. The catalog preserves current source dependencies. |
| R28 | Compile with instrumented flags and inspect spills | Applied to the evidence workflow. Five wrappers now return failed build/disassembly status and retain full logs. Runtime flags are unchanged. |
| R29 | Tune max registers and fast math | No blanket change. Require no-new-spill, numeric and latency evidence using the actual target build. |
| R30 | Run four Compute Sanitizer modes | Documented target checks. None ran here; tool support and ordering/progress proof remain separate gates. |
| R31 | Compare with FP32 and matching quantized reference | Apply with all mandated intermediate rounding and per-expert operations. A lone GEMM reference is incomplete. |
| R32 | Interleave A/B and log full environment | Apply. Historical different-seed/node outcomes stay descriptive; accept only the preregistered matched gate. |
| R33 | Use Triton for cheap schedule exploration | Optional distinct experiment. A wholesale backend rewrite has no earned allocation or speed claim here. |
| R34 | Hand PTX/SASS optimization after the main limiter | Conditional on compiled dependency and exposure evidence. Preserve required fences and numerical semantics. |
| R35 | Correctness, microbenchmark, end-to-end gates | Apply with cold graphs, complete call costs, all-six regression protection, independent controls and held-out routes. |
| R36 | Prioritized four-wave action plan | Reconcile with current priorities. D32 stays with its active worker; this intake neither creates another lane nor repeats completed experiments. |

## Bounded outcome and remaining work

The concrete implementation change is the compiler-status repair. It removes a false success signal, not GPU execution time. The reference library and automatically loaded instruction link make the other decisions recoverable after a context reset.

No new runtime change met an evidence gate in this audit. The strongest structural alternative remains a complete compatible grouped engine, compared with the selected persistent native path. Current Humming/FLUTE source studies leave specific numeric, work-reduction or net-cost gaps. A future allocation needs a new mechanism or matched evidence that resolves one of those gaps. This task supplies no basis to repeat their old configurations.

The cheapest next performance discriminator is an owner-run capture of the selected B4/MAIN hot loop and memory traffic when the existing experiment allows it. Measure issue gaps, compiled decode overlap and physical L2/DRAM traffic on real routes. Stop a proposed treatment if it changes no limiting dependency/work, adds spills or fails the complete-call/regression gate. Read [the current steering index](../../steering/README.md) first because experiment ownership and state can change independently of this dated digest.
