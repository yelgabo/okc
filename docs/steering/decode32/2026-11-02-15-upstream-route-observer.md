# D32-ATTN-05: bounded upstream routing observer

November 2, 2026 Pacific. Continues the [terminal D32-ATTN-04 routing investigation](2026-11-02-14-routing-order-diagnostic.md) under the user's direct execution instruction. The selected incumbent and attention treatment remain frozen. This is a private diagnostic, with no performance acceptance path.

## Exact source and question

[Observer payload](evidence/D32-ATTN-05/observer.py) SHA256 `a09982243ff3bd5139f5ba87acc1a10905dda58614ad21246e1cefe99894a324`, 936829 bytes, derives from frozen `9e5e5d64`. [Manifest](evidence/D32-ATTN-05/observer.manifest.json). [Generator and report reader](../../../tools/d32_route_observer.py), [CPU tests](../../../tools/d32_route_observer_test.py).

Hypothesis: the observed routing-order difference can be localized before top-k by comparing graph inputs, QKV, attention output, normalized router input and pool/router results. It may instead arise at selection or from private-state handling. The previous receipt does not distinguish these causes. This diagnostic changes observability, not the candidate arithmetic or routing rule.

The diagnostic reproduces only the observed T32 sliding input bank, seed1102031, across four independently owned placements. It checks one initial replay and up to eight fresh replays. Both treatment-bound placements must report the treatment inactive. The original exact route assertion and inherited numerical/cache/ownership guards remain.

## Observation and controls

Python wrappers forward the original arguments and return objects unchanged. During capture they retain QKV, attention output, normalized router input, raw pool/router results, actual local IDs and actual routing weights. They record the actual caller, projection pick and split/combined layout. The native loader record must identify selected B4 source `c7c201ae` and its B4 flag. Unknown layouts, missing boundaries and cross-owner probe storage fail closed.

Before replay, the diagnostic snapshots stream, attention input and K/V caches on CPU. After replay it snapshots the retained boundaries and graph data, compares every placement with control0, and compares each placement with its own preceding replay. It records exact byte equality and finite maximum differences. Initial-to-fresh changes and the first fresh cache update are distinguishable from repeated fresh replay, rather than treated as automatic instability.

On a route mismatch it records the differing rows, IDs, associated native weights and selected logits before invoking the unchanged strict failure. Split-layout logits are observed producer values. Combined-layout logits are reconstructed on CPU in the documented partial order from the retained raw producer tensor, and are labelled as such. They are not a bit-exact claim about Triton sigmoid or biased selection scores.

Retaining tensors and copying observations changes allocation and execution conditions. This source cannot establish timing gains. It emits one compressed `D32_ROUTE_OBSERVER` capsule, then fails setup at the original mismatch. If all eight fresh replays remain equal, it emits an inconclusive completion capsule and deliberately raises `D32_OBSERVER_COMPLETE_NO_TIMING` before timing. It cannot emit a completed matched-performance report.

## Local validation

All five real-CPU-Torch tests pass, including the complete newly generated prepare/capture/replay sequence. They cover unchanged forwarding, separate owner storage, replay-updated observations, combined-layout reconstruction, finite/nonfinite FP8 diagnostics, unsupported layouts, missing boundaries, an injected same-set slot swap and malformed report identity/selection/coverage/timing claims. [Complete output](evidence/D32-ATTN-05/cpu-tests.txt). CPU device operations are simulated; this is neither CUDA graph qualification nor target correctness evidence.

The initial regression reproduced the missing-boundary evidence in D32-ATTN-04 while preserving its strict failure. The observer now records that evidence and retains the failure. The generated CPU replay test also checks that the first fresh cache write changes the next pre-replay snapshot, while later identical fresh replays remain stable.

[Source identity capture](evidence/D32-ATTN-05/source-identity.json) verifies identical attention device literals and five native factory compiler inputs against frozen D32-ATTN-04. No device code or compiler flags changed, so a new device-only compilation would repeat those inputs. Runtime native selection is checked again on target. The generator reads the frozen payload directly; rebuilding through unrelated generators could embed host-dependent paths.

## One-run decision and stop

After publication, verify local/remote bytes and fresh shared jobs/quota. Submit exactly one private diagnostic when no jobs are live, persist its accepted ID before other measurement, and follow that own ID only.

If routes differ, validate the complete capsule and receipt, then identify the first observed cross-owner or same-graph boundary difference. Equal earlier boundaries narrow the next source investigation; a maximum difference alone is not a causal proof. Choose a correction from that evidence before another run. If this retention-altered diagnostic does not reproduce, classify it as inconclusive and stop this exact instrument. Do not repeat seeds, change the route gate, infer a fixed bug or claim a gain. A different evidence-backed observation or research design is required next.

The performance candidate remains unconfirmed. Any later promotion still requires matched C32 and all six loads, official correctness and cache/replay checks. CUDA Optimizer retains the independent B4 phase study. There is no separate qualified D32 kernel mechanism in this report.
