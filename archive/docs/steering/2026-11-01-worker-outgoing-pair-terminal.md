# Outgoing seam private check: terminal graph-query failure, allocation stopped

November1 Pacific / November2 UTC, 2026. Own private **9adfd4dbc98f4998**, muse_decode_layer/check, is **done / passed=false / returncode112**. The fixed allocation stops here: **no retry, second check, public/scored action, production edit or source rescue**. This is an invalid/incomplete measurement, **not a measured PDL loss or a closure of all outgoing-edge designs**. Zero new performance-qualified candidates.

## Exact ownership and frozen source

Preregistration **3d00a6211e3b65e8399e1fabca6f0e05061514ab** and acceptance **ddbed42cfa765307081dcd16309e78780f77f46c** were pushed and remotely verified before interpretation. [Registration](2026-11-01-worker-outgoing-pair-registration.md), [frozen manifest](evidence/2026-11-01-worker-outgoing-pair/manifest.json), [offline pins](evidence/2026-11-01-worker-outgoing-pair/offline-pins.json).

- Physical upload **f25d714e928835fba07741ca56fa8783137032f268652c22ec4f5511d771bc9a**,934942B; exact [compressed source](evidence/2026-11-01-worker-outgoing-pair/check.py.gz).
- Frozen owning reader **5435108e9eb729c7fe1f705244845cc7c916b9af780081e682c83ae81f736c1b** remains unchanged, as does the uploaded file.
- Server content identity **0af65ed333351fae1b386cac469aee043b3ac24b040cf6713c65fa9b76147ea2** and its payload_ref match the owning board/mode/team receipt; this identity is not the physical SHA256.
- H200 node **n-2bbs7xxbcpmk3emdvrn5jsqeic**, cluster sfo3, driver595.91.07; evaluator **1ba270d841be831b06a5862f4094cd0d5072edfbd0f6e84b95f79248296367dd**. Full evaluator file/image pins remain in the receipt. No cross-node timing comparison is attempted.

Accepted04:51:11.656UTC; started05:08:41.850UTC; finished05:11:58.703UTC. Queue delay1050.193567s, worker elapsed196.853185s. **prepare_s is null**: worker duration is not a measured prepare time or evidence of a295s guard failure. Own wait16147 exited1 and preserved complete raw stdout before parsing; it is terminal, not to be restarted. Terminal quota6private/5scored remaining,4checks used; only one check was dispatched by this allocation.

## What actually failed

The setup record reports `AcceleratorError: CUDA error: invalid device function`. The retained traceback is:

```text
prepare -> _srp_prepare -> _wl_prepare -> _wl_capture
    -> _os_graph -> query.seam_graph_edges(graph.raw_cuda_graph())
torch.AcceleratorError: CUDA error: invalid device function
```

This localizes the failure to the explicit graph inspection binding, **not to an NVCC/host compilation error**. It does not identify which internal CUDART call failed: the binding queries nodes, node types, kernel parameters and edge data, with several checked API calls but no per-call label in the retained exception. **The failing row/arm is also not logged**, so attributing this to adapterON, programmatic edge creation, the raw graph, cooperative incompatibility or GPU numerical work would be unsupported. The deterministic first row's attempted arm order is1/2/0/3, but that does not establish which row actually failed.

There is **no OUTGOING_REPORT capsule**, no official test record, no benchmark, no profile, and no score/CI. The frozen reader rejects this failed owning receipt with exit1 before any complete/performance gate can be accepted. Summary fields are **complete=false; private_raw_gate=null**, not a valid negative timing gate. [Owning receipt](evidence/2026-11-01-worker-outgoing-pair/own-terminal-receipt.json), [transport](evidence/2026-11-01-worker-outgoing-pair/own-terminal-transport.json), [reader rejection](evidence/2026-11-01-worker-outgoing-pair/frozen-reader-rejection.json), [compact terminal fields](evidence/2026-11-01-worker-outgoing-pair/terminal-summary.json).

## What target execution did establish, and its limits

The log reaches `G_RELEASE_STARTUP_OK ... 56 verified_replays` and then the private `WL_PREP ... outgoing_only_four_placements resources` line. This is more evidence than CPU mocks: target extension build/load progressed far enough to execute native expert and original/adapter secondary resource exports and reach private capture. There is no compiler diagnostic failure at the observed stopping site. However, the detailed actual build census and secondary resource values were emitted only in the final capsule, which was never produced. Do not manufacture those missing records or claim complete compiled-equivalence qualification.

Source-guided interpretation of the frozen control flow: the logged WL_PREP line is after all primary/common resource parity assertions, the secondary no-resource-loss assertion,32empty-native checks and16legal occupied-to-empty checks. Those preceding operations returned rather than raised. This does **not** substitute for the complete owning capsule or for full-layer fresh-output/timing qualification. Cleanup restoration has CPU/source coverage but no retained terminal success assertion.

Logged native resource vectors match all four placements and both expert entry tuples. These are direct log observations, not a causal occupancy analysis:

| Backend | Registers | Local bytes | Static shared | Dynamic shared | Active blocks/SM | Threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 145 | 0 | 0 | 214784 | 1 | 288 |
| tiny2 | 145 | 0 | 0 | 214784 | 1 | 288 |
| mid | 152 | 0 | 0 | 214784 | 1 | 288 |
| b4 | 96 | 8 | 0 | 231168 | 1 | 576 |
| main | 96 | 0 | 0 | 231168 | 1 | 576 |

The b4 local-memory value is shared by all controls/treatments; it is not evidence of newly introduced spilling. [Parsed log resource observation](evidence/2026-11-01-worker-outgoing-pair/native-resource-observation.json). Each private placement logged334995616owned mutable bytes; free memory134490292224B. These observations do not establish the graph seam, numerical full-layer equivalence, speedup, adaptation cost or isolated PDL effect.

## Decision, next ownership and evidence preservation

**STOP current allocation on the named runtime/instrument graph-query blocker.** No budget is used to repair/rerun the graph observer, change flags, weaken edge identity/edgeData, remove fences, introduce timing instrumentation, reopen RMW/packedRNE/gamma or switch to an unallocated fallback. The required net arm3-vs-each-raw threshold, raw-only duplicate null, short-only contribution and context regression rules remain frozen/unmeasured. The earlier15/15 CPU tests and captain's independent400-fixture arithmetic differential tests do not resolve this runtime failure.

If separately allocated, the smallest source/API investigation is to identify which graph-node query cannot represent this captured graph and establish an identity-preserving query compatible with its kernel representations. Driver/runtime kernel-function interoperability is a **hypothesis**, not a diagnosed cause; the present exception does not distinguish it. A future measurement would require explicit new authorization plus unchanged numerical/protocol/ownership gates, and cannot be called a continuation/retry of this one-check allocation. Broader expert-pipeline research remains captain/researcher-owned. README stays captain-owned.

Exact failed-run breadcrumbs are also retained as `muse_decode_layer/profiles/g-outgoing-pair.9adfd4dbc98f4998.{json,log,txt}`. JSON/log are raw owning artifacts; TXT is explicitly a synthesized failed-setup summary, **not a GPU profile**. Captain receives this terminal report plus the run-log/ruled-out wording: this exact query-enabled instrument is blocked; the outgoing seam has no valid performance result. Terminal archive manifest pins all owned archived files. No further execution is authorized by this report.
