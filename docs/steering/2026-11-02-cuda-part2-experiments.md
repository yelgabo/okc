# CUDA part 2: independent compiler and MoE experiments

November 2, 2026. Owner: OKC CUDA Optimizer. User instruction: "Be independent and run some experiments. I want to see positive results." Evidence base `f720ac617d9eb82f3bd69d16be2ff52809f64edd`. [Plan and prospective gates](../superpowers/plans/2026-11-02-cuda-issue-groups.md). [Research correction digest](../research/cuda-h200/part2.md).

Current outcome: both H200 comparisons pass correctness and replay checks but fail their matched speed gates. Constant scales remove local B4 spills; the corrected scale/group combination also reduces static group operations by 25%. Neither yields a measured GPU speedup. Both exact candidates stop, with the runtime baseline preserved.

## Scope and selected implementation

The supplement's snapshot `779b06ad` and standalone blob `96b5bf91433fc87154de1edc40ce7a31b5c4c025` are verified locally. Its conclusions correct packing, launch-residency and operand-orientation assumptions; it ran no GPU experiment. This optimizer instead tests the selected layer-native MoE body `c7c201ae`, from full control `ba9e327c`, through the already validated standalone four-state adapter.

D32 Worker's attention confirmation remains separate. This work edits no layer candidate, changes no live D32 experiment, and polls none of its jobs. Its private MoE measurements use that problem's separate quota. Public submissions remain outside this experiment.

## Experiment 1: shape-specific issue groups

Hypothesis: four k16 operations per WGMMA commit group for NTT<=4 can reduce collective issue overhead, using the accumulator capacity released by smaller N tiles. Retain two operations for wider tiles. The initial implementation intended to preserve deferred work's completed-k16 boundary; the later schedule audit below found that implementation wrong. Packing, arithmetic order, stage drains and source publication text remain.

Local CUDA 12.8.1 SM90a compilation completes for both bool entries in B4 and MAIN. The initial local-only gate stopped the treatment, a decision rule withdrawn in the correction below: B4 spill store/load bytes change from 4/32 to 80/88; MAIN from 0/0 to 4/24. Stop this exact treatment before target allocation. No GPU latency or general issue-group limit follows from this result. [Resource evidence](evidence/2026-11-02-cuda-issue-groups/group-resources.json).

## Experiment 2: constant scale multiplier

The WGS loop computes `(OKC_SC6W ? 4u : 16u) + p.zero`. The immutable host wrapper sets `p.zero = 0`. The original opaque value forces an integer multiply to a particular instruction pipe. Remove that dependency and let the current compiler fold the identical integer multiplier. This preserves reconstructed scale bits, BF16 multiplies, all numeric boundaries, grids, scratch and ordering operations.

The initial local compiled comparison removes B4's spill stores/loads and its eight static LDL plus one STL instructions per entry. Both MAIN entries retain zero spill traffic. HGMMA, wait0, wait1, warpgroup-arrive and async-fence counts match control per entry. This is a positive compiler result, not a measured H200 speedup. [SASS counts](evidence/2026-11-02-cuda-issue-groups/scale-sass.json) and [resources](evidence/2026-11-02-cuda-issue-groups/scale-resources.json) record the observations. The final source removes the stale inline annotation describing the control's opaque multiplier.

## Registered private measurement

[The exact payload](evidence/2026-11-02-cuda-issue-groups/paired.py) and [manifest](evidence/2026-11-02-cuda-issue-groups/paired.manifest.json) preserve source identity and the seven actual compiler requests. Reuse the existing complete harness's three banks, two route distributions, nine correctness sizes, six timed loads, four independent prepared states, fresh capture inputs, occupied/empty transitions, cold replay, reversed traversal, duplicate nulls and cleanup checks. Runtime and decision contract remain byte-identical to their established implementations.

One initial private check is authorized here. Refresh live jobs/quota immediately before it and record the accepted ID. Wait only on that ID. An official score compares the candidate with SGLang; treatment/control improvement comes from the matched capsule. Require all six bank/distribution summaries and four placement crosses to meet the existing >=1% equal-six factor and duplicate-null gate, with no >=1% affected-load regression, before promotion. One further confirmation is available only if the first result is promising. No automatic sweep after a loss.

That registration checkpoint is superseded by the terminal result below. No runtime candidate has been promoted.

## Evidence and remaining gates

Six focused CPU tests verify reversible source edits, group-bank reuse and arithmetic issue order, immutable host suffix, publication text and the scale transform's dependence on the verified host-zero assignment. The [baseline-aware hotspot map](evidence/2026-11-02-cuda-issue-groups/hotspots.md) identifies complete expert execution as the standalone target. Old profiling activity is not treated as exposed removable time.

The compiler driver discloses its host-tail extraction, declaration-only lookup adapter and compile-only references. Actual target compilation, numerical results, fresh replay correctness and latency remain required. The user-authorized experiment does not need a forecast of a win. Independent agent review is unavailable under the no-new-agents directive; no new memory-ordering protocol is being shipped.

## Private check accepted

Private MoE job `3aa3885f6b734f8c` accepted from source commit `ca588999`, payload SHA256 `ccef8a76df0d8bf0e60a882a6161c0d91925ac389743aa70160514b371f2dfd0`, 342733 bytes. The source and seven build descriptors are fixed. Fresh preflight showed 20 MoE checks remaining and only the separately owned D32 confirmation `4fece18b20a2447b` queued. The optimizer owned the sole waiter for its MoE ID and left that D32 job with its worker. [Accepted source/job record](evidence/2026-11-02-cuda-issue-groups/accepted.json). That acceptance checkpoint is superseded by the terminal result below.

## Compiler-only interaction result

The third source treatment combines the constant scale multiplier with an ordinary load of the launch-stable `S.ssp` pointer. Both selected builds compile without spills. Their complete cuobjdump SASS text is byte-identical to the scale-only build, including both bool entries and helper functions. [Hashes](evidence/2026-11-02-cuda-issue-groups/ssptr-identity.json). This produced no distinct local machine-code treatment. It received no GPU run as a budget choice, leaving target behavior unmeasured. It does not establish what another compiler would emit. The accepted scale-only job is unchanged.

The [loop audit](evidence/2026-11-02-cuda-issue-groups/scale-loop-audit.json) also limits the scale-only interpretation: most innermost 16-HGMMA loops keep their instruction counts. Static total instructions and removed spills alone do not forecast an endpoint gain.

## Constant-scale H200 terminal result

Own private `3aa3885f6b734f8c` passed five official tests, six benchmarks and all six profiles, without flags. Official speedup versus SGLang is 2.341016864x, 95% CI [2.305901611, 2.376132117]. This is not a gain over our control. The complete private harness passed 54 cases, 378 observations, 216 captures, 1,152 timed replays, 54 restorations and cleanup. Worst global/row relative errors were 0.000138487/0.000372057. Prepare took 273.985619 seconds. Queue and worker times were 626.302461 and 331.904116 seconds.

All 24 whole-six matched factors range from 0.992411 to 1.004932 and fail the frozen >=1% plus duplicate-null gate. Their descriptive geometric mean is 0.997953. Unchanged short paths also drift, so this does not establish a causal 0.2% slowdown. Mean descriptive T32/T128/T256 factors are 1.000066/0.999202/1.001283. Three affected bank/load cells regress at least 1%. Stop the scale-only performance candidate, with no confirmation or runtime-default change. The positive compiler result did not translate to a measured speedup. [Owning receipt](evidence/2026-11-02-cuda-issue-groups/status-wait.raw.json), [reader analysis](evidence/2026-11-02-cuda-issue-groups/analysis.json), [compact summary](evidence/2026-11-02-cuda-issue-groups/summary.json), and [complete profiles](evidence/2026-11-02-cuda-issue-groups/profiles).

## Group-schedule correction and coherent alternative

The first group-only compiler treatment and the first combined scale/group treatment used equality against the original completed-k16 threshold. Active native `c7c201ae` defines FCI=4 and GS=2, a threshold of ten operations. A nearby helper defines FCI=3. Four-step groups cannot equal ten, so the flawed small-NTT branch skipped its first-stage deferred flush. SASS exposed eight missing global async fences. Neither grouped treatment reached a GPU. Their spill observations apply only to those exact flawed sources, not a correct implementation of the intended schedule. The accepted scale-only GPU payload is unaffected.

The corrected coherent alternative rounds the threshold up to the first group boundary: twelve operations for four-step groups, ten for two-step groups. Ten CPU tests now include the active source constants, reachability at every original group boundary, operand-bank lifetime, source reversibility and receipt/source rejection. Root instructions load this correction automatically.

This delay introduces no new dependency before the flush: both ring-full waits precede the first operation of the already acquired stage, and the extra two operations use only that stage's shared activations and loaded weights. The weight producer may wait for slot release, but the consumer requires no new producer publication between steps ten and twelve. Stage wait0 and activation-slot publication, segment wait0, all phase-end flushes and existing release/acquire operations remain. Alternating register banks are reused only after wait1 completes their earlier group. The required register ordering and group-completion rules are in [PTX ISA 8.7, WGMMA proxy operations](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations). Source reasoning does not replace target correctness checks. Compile the corrected combination before deciding whether any route earns a new bounded private check.

## Corrected combination registered

The corrected scale/group candidate native `7db5ec77` compiles without stack or spills in B4 and MAIN, both bool entries, at 96 registers. Per-entry HGMMA512, wait0=48 and async-fence84 match control; wait1/warpgroup-arrive decrease 256 to 192. All publication operations remain represented. [Resource and SASS audit](evidence/2026-11-02-cuda-issue-groups/group_scale_fixed/audit.json).

The [new exact payload](evidence/2026-11-02-cuda-group-scale/paired.py) is 342999 LF bytes, SHA256 `c4e6d1088fdc0f8d61a3547d9d01c6cdabc42804cefac39741622645f5feade9`. Its [manifest](evidence/2026-11-02-cuda-group-scale/paired.manifest.json) records all seven actual compiler requests. Runtime/contract and six-load matched gates remain unchanged; this is a combined B4/MAIN treatment, not a posthoc per-load rescue of the scale-only result. One initial private check and a confirmation only if promising. At registration, fresh preflight showed 19 MoE checks left and separately owned layer check62a4f6b27ed74b50 running. Its owner retained that job. The accepted identity and terminal result below supersede this registration checkpoint.

Accepted own private job `3c45788127e945c4`, source commit `9669ee15`, after remote payload-byte verification and a fresh jobs/quota check. The optimizer owned its sole poller, which finished at terminal. [Accepted identity](evidence/2026-11-02-cuda-group-scale/accepted.json). The terminal result below supersedes the accepted checkpoint.

The additional local build includes the payload's `-lineinfo` flag, which the initial compiler driver omitted. All four instruction-encoding sequences match their earlier local build exactly; only disassembly header metadata differs. Zero spills and the static collective/publication counts are reproduced. Full compressed SASS, logs, flags, hashes and differences are archived in [the flag-aligned evidence](evidence/2026-11-02-cuda-issue-groups/group_scale_lineinfo). The actual H200 compiler and complete captured replay remain the decisive checks. The offline reader resolves archived logs relative to the evidence folder while retaining the receipt's original path in its attestation, so a different checkout can replay the audit.

Interpret the corrected candidate as one coherent treatment: scale folding, small-NTT issue batching and its two-operation flush delay interact. A matched gain would belong to that complete treatment. Static group counts do not isolate its causal contribution, and this experiment does not require each ingredient to win independently.

## Combined treatment H200 terminal result

Own private `3c45788127e945c4`, source commit `9669ee15`, passed five official tests, six benchmarks and all six profiles without flags. It passed the complete 54-case, 378-observation, 216-capture, 1,152-timed-replay, 54-restoration and cleanup checks. Worst global/row relative errors were 0.000135982/0.000171862. Prepare took 276.366447 seconds; queue 545.209209 and worker 349.580438 seconds. Official versus-SGLang speedup is 2.348334366x, 95% CI[2.313109350,2.383559381]. This is not a paired improvement over the control or the previous job.

The unchanged strict reader validates the source/manifest, seven compiler requests, owning receipt/log and complete coverage. All 24 whole-six factors range 0.995560..1.008180; none reaches 1.01. The descriptive geometric mean is 1.001063, within the instrument's control spread. Mean descriptive T32/T128/T256 factors are 1.002055/1.001833/0.999788. Three affected bank/load cells regress at least 1%. All six bank/distribution groups fail the frozen gate. [Owning receipt](evidence/2026-11-02-cuda-group-scale/status-wait.raw.json), [strict analysis](evidence/2026-11-02-cuda-group-scale/analysis.json), [summary](evidence/2026-11-02-cuda-group-scale/summary.json), and [complete profiles](evidence/2026-11-02-cuda-group-scale/profiles).

Stop this exact corrected combination without a confirmation, dispatch-mask rescue, threshold sweep or runtime promotion. It passes correctness but supplies no convincing endpoint gain. This scopes the loss to native7db5ec77, these B4/MAIN flags and this complete standalone workload. It does not reject every WGMMA batching design or establish a hardware ceiling. The source, regression tests, compiler evidence and negative GPU results remain available for later work.

## Decision and remaining uncertainty

Four compiler treatments and two private H200 comparisons produced a useful compiler improvement and no promoted runtime speedup. The current two-step baseline earns retention against the two remotely tested alternatives. The local-only pointer treatment remains unmeasured on the target. The experiments remove a stale instruction-placement assumption and test a coherent change in matrix-loop scheduling, rather than only lowering a register count. Their measured endpoint factors remain near 1.0. They do not show whether decode issue, startup, handoff or tail time limits the selected path.

The next useful discriminator is a bounded phase trace of the selected complete MoE: measure startup, steady stages, A/B handoff and finalization, with an uninstrumented matched control to price observer cost. A bare bandwidth ceiling, profiler activity sum or static instruction count cannot supply those exposed times. Choose another implementation only after that evidence identifies removable work, or a complete stronger comparator supplies a distinct source-backed mechanism. This is a next gate, not a new dispatch or assignment. D32 and ring-depth ownership remain unchanged.

Final verification: 22 focused CPU tests pass after the final reader/compiler changes; research coverage and pins pass. The constant-scale owning reader also reproduces its rejection from an independently copied evidence folder. Full disassembly and raw profiles are preserved. An independent reviewer was unavailable under the no-new-agents directive; these changed source proofs and conclusions received author checks only. Neither local compilation nor numerical agreement establishes a complete new memory-ordering proof.

## Environment fidelity clarification

The user asked whether local tests replicated H200 one to one. They did not. Local builds use the Linux ARM64 `nvidia/cuda:12.8.1-devel-ubuntu24.04` container and an SM90a target. They compile an extracted device prefix with a declaration-only adapter and compile-only launch references, rather than the complete PyTorch extension. The selected native source, key defines, O3 and lineinfo settings are recorded, but these matches do not establish environment parity.

The owning remote receipt identifies a different judge image, NVIDIA H200 hardware and driver580.159.03. It does not provide the remote nvcc/ptxas/PyTorch versions or the complete resolved extension build command. Those versions and local/target binary parity were not established. Local register, spill and SASS observations therefore cannot be asserted as target resource measurements. The two private checks separately compiled and ran the complete payload on the actual remote H200 environment; their correctness and timing conclusions do not depend on a local environment replica. No new GPU job is needed to clarify this evidence boundary. Root instructions now load this distinction automatically.

At this clarification checkpoint, the global research checker flags concurrent edits to `docs/ruled-out.md`, including archive-link updates and newer unrelated experiment closures. Its hash was not refreshed by this narrow environment audit. The environment statements above were checked directly against the compiler driver, archived compile flags and owning H200 receipt.

## Correction: local results cannot decide target performance

The user identified a decision error: treating local compilation as a hard rejection gate depended on unverified correspondence to the H200 build. That gate could discard a target winner. The compiler toolchain and wrapper were not matched one to one, and even target-confirmed spills can be outweighed by useful work reduction. The original no-new-spill rule and local-only code-identity closure are withdrawn as target-performance decisions.

Reassessment of this task's cuts:

- Initial group-only and first combined prototypes have an unreachable deferred-flush condition. That source-correctness defect independently blocks those exact sources. Their local spill counts establish neither H200 resources nor latency. Correctly scheduled group-only batching has no target result.
- Scale folding plus ordinary stable-pointer read has identical local SASS. Its target code and performance remain unmeasured. It was deprioritized for low expected information per GPU check, not disproved or verified as a duplicate H200 binary.
- Scale-only own3aa3885f6b734f8c and corrected combined own3c45788127e945c4 were compiled and tested remotely. Their failed matched performance gates remain valid within the recorded workload and controls.

Future local resource/codegen checks provide ranking signals. Source correctness remains a separate gate. For a promising candidate cut only by local resources, obtain the target compiler/build details or perform a bounded target compile or private measurement before asserting target rejection. Label a budget deferral as untested. Do not require every candidate to consume GPU quota or automatically reopen unsafe source. Current B4 phase-attribution ownership and other owners remain unchanged; this correction dispatches no job.

The root instruction, local-only closure rows and prospective plan now reflect this rule. Broader local-only closures elsewhere in the repository have not been re-audited by this narrow correction.
