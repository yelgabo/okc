# Expert harness real-Torch host validation: bounded orchestration PASS, payload held

November 2, 2026. Test sub-agent; no children. Allocation/source baseline: **1f59b9657f3da57a669543535d8da2010e82745a**. Implementation and first final evidence: **ca945bedc6e56e3a0bb561621a237afc60cfb9f8**. The accompanying publication commit removes two trailing blank lines from the test and refreshes its final receipt; no additional behavior change. This work stops within the 45-active-minute allocation.

**Disposition: PASS for the complete generated wrapper/private orchestration under the explicitly listed CPU facades; NOT a complete production-preparation or target qualification.** The test executes the entire newly generated module, its actual final prepare entry, and its actual private loop. The production hold remains true in the generated artifact and is opened only in each isolated test namespace. No dispatch_ready claim, hold removal, candidate/pair/default edit, install, policy change, GPU/CLI POST, target compiler, performance treatment, or public action occurred. Captain retains dispatch ownership. No judge job exists for these host runs.

## Evidence and changed identity

Repository/checkout: yelgabo/okc, branch t3/torch-host-validation, based on parent1f59b96. Read the coordinator, README, steering index, original harness handoff including its final environment correction, and the runtime/source artifacts before implementation. The previous default-interpreter/Windows-policy failures are not re-used as a blocker: the existing WSL interpreter imports **Torch2.10.0+cpu**, with CUDA unavailable, and performs the actual tensor operations used here. No environment installation or policy bypass was used.

- New test: [tools/test_expert_top100_harness_torch.py](../../tools/test_expert_top100_harness_torch.py).
- Only runtime change: [tools/expert_top100_harness_runtime.py](../../tools/expert_top100_harness_runtime.py), failed-prepare guard cleanup.
- [Final test output](evidence/2026-11-02-top100-expert-host-validation/final.txt): **13 tests PASS**, including multiple full-run and failure subcases.
- [Final machine receipt](evidence/2026-11-02-top100-expert-host-validation/final.json): exact test SHA, generated source/core pins, all four complete CPU reports and explicit synthetic event labels. This is the authoritative current receipt.
- [Inherited tests](evidence/2026-11-02-top100-expert-host-validation/legacy-tests.txt): **15 harness + 3 pair-package + 4 prototype = 22 PASS**; 35 test methods total across the four suites. These 22 retain their previous source/facade scope.
- [Before-fix output](evidence/2026-11-02-top100-expert-host-validation/before-fix.txt) and [receipt](evidence/2026-11-02-top100-expert-host-validation/before-fix.json): all three then-configured positive full runs completed, but eight failure subcases exposed retained guard registrations. The final suite independently reproduces the defect using the untouched old archived payload. Initial.json is a test-adapter import-protocol bring-up failure, not a candidate result; after-fix files are the intermediate 12-test run, superseded by final.
- [New held payload](evidence/2026-11-02-top100-expert-host-validation/harness-held.py): **918522 LF bytes**, SHA256 **4cdd4268d394c3b48a9549c201dde80a8e343a2335e455684fd7ea1cb0560a23**. [Manifest](evidence/2026-11-02-top100-expert-host-validation/harness-held.manifest.json). Core changed from **721dd475176e2a602ac6d717254772e3790b77c3cafd80d4f96fb9f9830828b6** to **881ad4521709299f1dcf2620600bb5c7e4e56848ef8bc0615e19110a5afb08d7**, solely for cleanup. The generator's conservative STOP_CPU_HOST_GATE/remaining labels are retained, not silently upgraded.
- Old held payload **c6f4c25d0b059e0893225515868c9280b3bcb6434692e95b8ed931b9f5c01402**, its manifest and earlier report remain unchanged. Exact control **ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab**, candidate **dff221ae6ccc2de35d7a4c123aea3e3eb1345d3685f6eb97e52cbbfef6a0c8c4**, pair prefix **4d1a4b2cdfd1a737bf7388deaafcc27900729ae71314c5911e8d66423afe3e23**, selected native sources and flags are unchanged.

Generated source is archived byte-for-byte, including inherited whitespace. Raw unittest failure output is also retained verbatim. Whole-artifact whitespace checks therefore report existing embedded comment whitespace and one raw-log trailing space; these are not normalized into a different source pin. Changed handwritten code has no whitespace-check findings after the final EOF cleanup.

## What actually executed

Every positive run enters the newly generated prepare, builds/selects native modules through the actual loader/selector against an inert compiler endpoint, makes four private placements with the actual storage-cloning implementation, runs transitions, all workload loops, captures, replays, checks, timing loops, decisions, serialization and cleanup. No _eh_private, _wl_clone, _wl_capture, _wl_check, _eh_routes, _eh_transitions, _eh_component, _eh_back, _qual_oracle, contract, or cleanup implementation is replaced in the positive tests.

Four complete runs cross caller inference mode OFF/ON with externally inference-created weights OFF/ON. **Each** visits all36 rows (six loads, two contexts, three frozen banks), four placements, 144 initial checks, 1152 fresh replay checks, 144 restores, and96 empty/occupied transitions. Each executes **8592 actual CPU facade replays**, while the native Python facade is called only624 times. Replay therefore is not a Python callback to the layer/native helper: captured ATen operations run again on real tensors and update the original captured output buffers. Returned outputs are inspected after replay, including the final timed replay. Mutation of the captured input before replay changes the outputs; a full-module mutant deleting the pre-replay restore is rejected by the unchanged generated oracle.

Real Torch operations include untyped_storage, data pointers/intervals, byte views and set_, shape/stride/storage-offset preservation, version counters, clones, in-place copies, BF16/FP32 conversions, finiteness, reductions, topk, cache indexing and content hashing. The actual owner checks see four disjoint mutable storage sets (208 bytes per small fixture owner), shared inventoried read-only weights, private nested Lt workspace/policy sets/module dictionaries/attention scratch, and preserved aliases within each owner. A strided alias mutation is visible only in its own owner; deliberate cross-owner sharing is rejected.

Generated _wl_refs chooses busiest/second/least group from each fixture's own input-dependent reference routing, with bank-distinct IDs rather than a fixed routing replacement. The actual modulo256/local32 mapping, distinct-ID and source-selection checks execute. This is natural routing **of the deterministic fixture**, not a reproduction of production routing distribution. Negatives cover corrupted local IDs after a numerically correct native call, duplicate IDs, wrong selected-source digest, shared output storage, first-empty nonzero output, initial and final replay output corruption, scratch corruption, opaque owners and native/replay exceptions.

## Explicit facade boundaries and limits

The following are deliberately substituted, not tested production implementations:

1. _EH_PREPARE returns a small representative prepared state. The incumbent's multi-stage production preparer, autotuning, packing and complete real prepared-state inventory are **not executed**. This is a material remaining scope limit, not an inferred PASS.
2. _ab_inputs uses small four-column layer tensors and short CPU KV banks, preserving six token counts, two contexts, bank seeds and fresh-input structure. It does not allocate the production80000-token context banks.
3. _WL_KERNEL, _ab_ref_front and _EH_BACK perform deterministic fixture mathematics. The layer/reference front shares fixture code, so agreement is **not an independent model-numerical proof**. The actual _qual_oracle, its adaptive-calibration/control logic, routing validator, comparison routines and KV checks execute against those substituted fronts/results. Actual production attention/expert/FP8 math is untested.
4. load_inline is inert. Triton declarations are inert and decorated device entry invocation is forbidden. The actual factory/selector processes pinned sources/flags but no native compiler or device code runs.
5. CUDA streams/synchronization/admission/events are facades. Capture records ATen operations; replay re-executes them with extra CPU allocations/copies. It does not establish CUDA graph address/lifetime/ordering/progress semantics. Memory preflight sees an explicit synthetic64GiB free figure; the256MiB flush is represented by256 CPU bytes. Events return **100 synthetic microseconds for every arm**. The frozen decision correctly rejects this null experiment in every bank; it is not a speed measurement. The strict target capsule reader rejects the CPU reports at its readonly-device requirement.

Thus this is substantially stronger host control-flow/storage evidence than the old helper mocks, but it must not be described as a production-preparer, kernel, GPU graph, admission, timing or full numerical gate PASS.

## Concrete defect fixed and regression

Previously guard_mark/guard_ids were initialized only **after** the normal preparation callback and its first budget check. A callback allocating guard-owned tensors then raising, or an immediate post-prepare budget failure, left both global registries retaining those tensors. Failures inside the private run also retained newly created official-state guards despite no state being returned. Reporting failure after the finally block had the same issue.

The repair snapshots entry guards first. Any prepare/private failure rolls back to entry registrations; successful private validation retains only the returned official state's guards. A final budget, serialization/report-size or output failure rolls back the newly allocated official guards as well. Prior caller guards remain intact. The existing restoration of module hooks, reference cache/dequant cache, build-observation map and TF32 setting is preserved. Successful official-state guard retention is checked separately; indiscriminately dropping those guards would be wrong. This is a host instrument cleanup fix, not a change to native memory ordering or candidate behavior.

The original archived payload still demonstrates the leak. The new payload passes prepare-failure, post-prepare budget, midrun budget, native/replay/guard/oracle failure, final budget and report-overflow cleanup tests with the actual generated entry. No public/job receipt was manufactured.

## Inference and reference-cache scope

Inference-created weights and prepared scratch really lack usable version counters; the test does not invent or patch _version. The generated private context explicitly disables inference mode, making cloned private scratch and internally built reference tensors ordinary versioned tensors. All four mode/weight combinations complete. An external inference-created ID tensor is accepted because the cache clones it under normal mode; the stored ID and returned reference tensors are versioned.

With real tensors, equal IDs reuse the cache; changed IDs and ordinary front mutation through a view recompute; a changed weights owner, mutated cached ID snapshot or mutated cached result rejects. An **inference-created front**, or an injected inference-created cached result, still raises 'Inference tensors do not track version counter'. This is deliberately fail-closed and is not general inference-front support. Immutable same-owner weights remain an assumption; no claim is made that out-of-contract weight mutation or version-bypassing writes are detected. Actual production-preparer/reference alias inventory remains a separate gate.

## Reproduction and exact test inventory

From this checkout in WSL, using only the existing interpreter:

~~~sh
/root/.cache/okc-host-validate/bin/python tools/test_expert_top100_harness_torch.py --evidence docs/steering/evidence/2026-11-02-top100-expert-host-validation/final.json
/root/.cache/okc-host-validate/bin/python tools/test_expert_top100_harness.py
/root/.cache/okc-host-validate/bin/python tools/test_expert_top100_pair_package.py
/root/.cache/okc-host-validate/bin/python tools/expert_top100_test.py
/root/.cache/okc-host-validate/bin/python tools/expert_top100_harness.py --out docs/steering/evidence/2026-11-02-top100-expert-host-validation/harness-held.py
~~~

The13 new methods are: actual_facade_replay_not_python_callback; archived_payload_reproduces_guard_leak; budget_failure_after_preparation_cleans_guards; final_budget_and_report_failures_rollback_official_guards; generated_entry_all_rows_and_inference_contexts; midrun_negative_cases_cleanup; missing_pre_replay_restore_is_detected_by_full_oracle; natural_routing_and_local_id_negatives; original_hold_prevents_all_device_facades; prepare_failure_cleans_new_guards; reference_cache_real_versions_and_inference_front; selected_source_negative_runs_generated_entry; storage_clone_aliases_private_policies_and_owner_negative. All have the test_ prefix. Their exact names and outcomes are in final.txt.

## Priority, next owner and stopping criteria

1. Parent/captain: review and integrate the narrow cleanup repair plus its held source pin. Do not repeat the previous dependency investigation or substitute the old15 helper tests for this new evidence. This allocation ends here; no new worker, continuation or dispatch is authorized by this report.
2. Treat the real production preparer's complete state/packing/inventory and original reference pathways as **open**, not covered by the small fixture. If the required host gate includes those pathways rather than only the generated wrapper/private orchestration, the full gate remains incomplete; obtain a separately bounded allocation rather than silently broadening this PASS.
3. Only an explicit later parent decision may remove the offline hold and freeze another payload/manifest. Before any permitted target job, reconcile live quota/ownership and retain exact source identity. No live quota was queried in this host-only task.
4. Target compiler/build,295-second complete preparation fit, CUDA ordering/progress and graph replay, full oracle/edge-value numerics, resource/admission and timing remain unmeasured. Keep the standalone adapter worker's ownership separate. A setup, identity, ownership, numerical, replay, progress or budget failure stops; no synthetic CPU timing can rescue it.
5. Frozen all-bank, whole-six >=1% gain, gain above twice the independent-duplicate null, held-out banks and affected-row regression gates remain unchanged, as do release/external-source obligations. No measured performance improvement or candidate-family rejection occurred; no runs-log or ruled-out technique-loss row is appropriate.

