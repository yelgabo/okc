# Why recent G experiments keep failing: independent receipt and test-contract audit

November1 Pacific / November2 05:39UTC2026. Evidence checkpoint `1a20b313dd8f64590a53efaca1dc9162f7818bda`, current main in C:/Users/gretc/Documents/GitHub/okc, origin https://github.com/yelgabo/okc.git. User asks why failures are frequent. Read current AGENTS/coordinator, latest steering and subsequent captain decisions, exact failed source/test boundaries and fresh owning receipts. No kernel/default change, new check, subagent or direct chat message in this audit. Current worker remains sole implementation/dispatch owner.

## Answer and accounting

**The recurring preventable weakness is the integration of our experiment setup: fixtures, preparation work and newly added GPU inspection APIs. CPU tests often replace the very behavior most likely to break.** We then spend a scarce judge run discovering that gap before reaching the performance question. This is shared implementation/review/allocation responsibility. Native kernel correctness, valid negative performance results, offline compiler screening and service failures must be counted separately.

I independently retrieved six recent relevant G jobs and verified job/board/team/mode/result fields: **three setup failures in our uploaded setup, one refunded external infrastructure failure, and two complete evaluations**. This is an explicitly named six-job cohort, not a statistical claim about every campaign run or every worker action. The three setup failures yielded zero official tests, benchmarks, profiles or score. The two complete evaluations each passed13official tests and produced12benchmarks/12profiles; neither established a promoted improvement.

| Owning job | Actual result | What went wrong / what was learned |
|---|---|---|
| `1fad70bbe3854cbc` | infra_error, refunded=true, no measurements | External judge/service failure. Do not attribute it to kernel quality. |
| `3fd1317143304bd5` | Setup112, GQUAL_PREPARE_BUDGET, no score | Our expanded qualifier failed its295s guard after initial native startup success. Reference/validation work was not proven to fit; missing phase breadcrumbs prevented pricing the failing stage. |
| `c1931f0c9e834485` | Complete PASS, prepare109.269811s | The inference-compatible/reference-reuse qualifier repair completed. Its2.121241x historical score is descriptive, not a matched RMW effect or promotion. |
| `16b4db27421d4714` | Setup112, CUDA719, no score | Invalid occupied-route fixture found: it repeats one expert in all8 top-k slots. Source/CPU modeling shows unsupported tiny-route multiplicity atT8. Exact failing GPU instruction was not localized. |
| `c5e4f89672f743bb` | Complete PASS, prepare178.253238s | Corrected legal fixture gives a valid paired RMW result: about-0.18%..+0.28%, below noise/materiality gates. This is an unsuccessful optimization measured successfully. |
| `9adfd4dbc98f4998` | Setup112, invalid device function reported in graph-query binding | Newly added captured-graph inspection did not establish native compatibility before use. Exact inner API/node/arm was not retained; no overlap timing result exists. |

[Fresh six owning receipts and summary](evidence/2026-11-01-failure-pattern-audit/cohort.json); [offline reconciler](evidence/2026-11-01-failure-pattern-audit/analyze.py), [verified classifications/source facts](evidence/2026-11-01-failure-pattern-audit/analysis.json). Reconciler asserts failed setup-only versus complete13/12/12 outcomes and the refunded service receipt. No numeric outcome is inferred from CLI success wrappers or a report's prose.

## Defect1: the routing mock omitted the domain contract

Actual production routing selects distinct top-k expert IDs per token. The failed paired fixture filled all8 slots with expert0, including repeated occupied transitions. AtT8 this assigns64entries to one expert, while legal distinct routing permits at most8entries for that expert. The selected Tiny2 path can therefore receive a fat job its tiny-only design rejects. This is a demonstrated invalid fixture, although asynchronous719 reporting prevents proving that exact trap was the observed instruction.

The old `Native.okc_run` CPU stand-in in `tools/proj_g_short_rmw_host_test.py:30` merely zeroes the output and records a replay operation. It does not check expert multiplicity or unique IDs. A full host-prepare test through that stand-in can pass while giving the real native kernel an impossible input. Host orchestration coverage did not equal input-contract coverage.

**Independent reproduction during this audit:** both current actual-entry regressions pass: restoring the old duplicate fixture is rejected, and a below-capacity duplicate is independently rejected. [Exact command/result and limits](evidence/2026-11-01-failure-pattern-audit/independent-regression-observation.json). These are CPU tests with native math mocked. They establish that a small legal-input contract check could catch this defect without GPU quota. They do not reproduce the original device719 instruction. The successful corrected c5e evaluation is the target evidence that the repaired experiment can complete.

## Defect2: graph tests mocked away the failing integration boundary

The frozen query `tools/g_outgoing_seam_query.py:21` traverses every kernel node, calls the runtime `cudaGraphKernelNodeGetParams` and returns its function pointer; it also queries nodes/types/edges. In `tools/test_g_outgoing_pair.py:83`, `Query.seam_graph_edges` simply returns preconstructed `graph_rows(address-1000)`. No C++ query, CUDA runtime/library interaction, target extension link or captured foreign kernel is executed. The negative exception test checks cleanup after an injected error; it cannot prove the real graph accessor works.

This explains how15CPU tests and400 synthetic score-oracle cases coexist with a native setup failure: they answer different questions. The score mathematics tests are useful; they never exercised the failing boundary. Native startup56replays and logged resource parity also answer different questions from mixed full-layer graph inspection. The failure is not evidence that every expert kernel is numerically broken.

**Most plausible specific explanation, not a proven cause:** runtime host-symbol versus driver-function identity in a graph containing kernels from different libraries/runtimes. NVIDIA's CUDA13 graph accessor explicitly permits invalid-device-function and previous asynchronous errors; its interoperability documentation supplies the explicit cudaGetFuncBySymbol/cudaFunction_t/CUfunction bridge. These facts make the identity hypothesis concrete but cannot identify the exact failing call/node/arm from the old log. Primary references reopened in this audit: [CUDA13.0 graph API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__GRAPH.html), [CUDA13.0.2 runtime/driver interoperability](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__DRIVER.html). No claim that the documentation itself diagnoses our job or guarantees the proposed repair.

The important second defect is failure observability: API/code/node/row/arm were absent, and secondary-resource/build records were saved only in a final capsule that never printed. We know where Python reported the failure, but cannot reliably name the native call or exclude a delayed earlier error. That turns one failed check into another investigation rather than a quick, attributable decision.

## Defect3: preparation cost and caller context were weakly modeled

The original all12 qualifier combined substantial reference computation, fresh/empty replay and validation before returning within295s. Its failed guard is real; prepare_s and detailed phase timings are missing, so worker wall duration is not a measurement of prepare cost and cannot identify a dominant stage. The later qualifier completed109.27s after a concrete reference-reuse/context repair. This proves a feasible setup exists, not the isolated speed contribution of each repair.

An intermediate cache used Tensor._version under inference_mode and was rejected offline before submission. Another source review caught90 versus90a factory flags before the outgoing check. These were successful reviews, not additional judge failures. They show why tests must match the real caller context, selected factory and data contract rather than merely make expected literals agree. They also show that unit/source checks are valuable when they target the right assumption.

## Underlying process and environment gaps

1. **We treated source/mock coverage as a stronger readiness signal than its scope warranted.** The dangerous native boundary remained unknown while test counts grew. The existing AGENTS principle7 already requires validating the instrument; the missing step is enforcing it against the actual risk.
2. **Small performance treatments accumulated substantial new setup/observer machinery.** A new fixture, state wrapper, four-placement mapping, graph API and native-identity query add independent ways to fail. Prefer a stable completed runner and the minimum integration necessary for the performance question. This is not permission to drop correctness/ordering/edge gates.
3. **Our local loop does not currently execute the full target integration.** The documented local setup supports CPU Torch and device-only compiler screens; full target extension host compilation/linking and H200 captured execution were not established. The judge becomes the first native integration run. That is a real constraint to budget explicitly, not an excuse to call mocks equivalent to target proof. Successful symbol resolution alone does not execute an API or compile/link the extension.
4. **Even the completed experiment targeted a narrow uncertain gain.** RMW's valid subthreshold result is useful closure, but its setup repairs consumed substantial campaign attention. Candidate selection and implementation reliability are separate weaknesses; perfect harness execution alone does not supply the several-percent improvement needed.

No evidence supports blaming the model/provider for these particular judge setup defects, claiming the whole production kernel is broken, or counting every offline resource rejection as wasted GPU quota. The retired chat's provider failures are a different incident and outside this six-job count.

## Concrete correction and current ownership

The goal is fewer untested boundaries and more informative results, not more generic tests or another long checklist.

- **Contract-first regression:** every repaired harness defect must have a negative that rejects its exact old failing behavior at the generated entry point. Preserve valid distinct-topk fixtures, inference context, replay ownership and selected compiler flags. Routing now has these negatives; retain them.
- **Name the unverified native boundary:** before dispatch, state which actual build/link/runtime/capture assumption is being tested for the first time. Mock, header, symbol-resolution, compilation, target correctness and timing evidence must remain separately labelled. Exercise the narrow representative integration early in the same allocated check and emit phase/resource prefixes plus named API/node/row/arm diagnostics before broad timed work. Do not require another standalone observer GPU job or rebuild an entire harness merely to improve reporting.
- **Keep one fixed experiment and a terminal gate:** a repair needs a concrete changed assumption, minimal source delta and an owner. A valid timing loss closes the optimization; another instrument failure closes the lane under current allocation. No automatic quota-funded repair chain, new seed, weakened gate or threshold rescue.

Current captain decision `1a20b31` already separately allocates one fixed integrated driver-query repair, <=25minute ready-or-blocked checkpoint then one conditional private check. It preserves the failed upload/reader, adds canonical identities/driver APIs/explicit linkage and prefix diagnostics, and closes this seam campaign on another instrument failure or valid subthreshold result. The scratch6CPU tests and local symbol lookup are not target validation. I have not launched that check or messaged the external chats; this audit neither duplicates nor expands its authorization. Worker owns integration/receipt, captain owns allocation/ROI, and broader candidate discovery must continue without treating a repaired observer as a new performance candidate.

Independent audit validation: six owning receipts re-fetched and reconciled, two specific routing regressions rerun successfully, graph mock/native source boundary inspected, primary API contracts checked, and existing repair/allocation reconciled. No new GPU quota consumed. Remaining unknown: exact9ad inner native failure and whether the fixed seam is fast enough. These require target evidence; this report does not claim them solved.
