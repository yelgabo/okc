# Single-arm all12 preparation: deadline failure and bounded reference reuse

Evidence clock: 2026-11-01 23:13:12 UTC. This is a new preparation-only source epoch, not a kernel optimization, completed H200 qualification or performance result. The accepted job and failed source remain unchanged.

## Owning failure

Root's owning `3fd1317143304bd5` receipt (public layer `run`, source `97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7`) is terminal **setup fail**, `GQUAL_PREPARE_BUDGET`. It printed successful native G56/56 verified expert replays, then reached `_gq_private` → `_gq_empty` → the 295-second guard. There are no correctness-test, benchmark or profile records and no score. GPU H200, driver 595.91.07, node `n-2bbs7xxbcpmk3emdvrn5jsqeic` are receipt observations, not preparation phase measurements. Root owns the canonical receipt/archive and run/closure rows.

The failure location does **not** identify the cell ordinal. Thus it does not prove the first empty graph failed, nor that reference arithmetic, cloning, compilation or any one component consumed most of the budget. No measured per-phase times exist in that receipt. The source order before each empty entry is normal prepare/G56, private setup if first cell, then base and fresh input/reference-front creation. A later-cell failure also includes all preceding cells' validation.

## Source cost inventory

* Normal prepare and G56 remain intact; compilation/tuning and the inherited validation precede all12 qualification. Blocking native preparation cannot be interrupted by a Python deadline guard.
* `_wl_readonly` already shares inventoried input/packed weights and immutable lookup storage. `_wl_clone` clones mutable backing storages once with alias memoization; the existing ≤2 GiB mutable-byte gate and free-memory guard remain. A private-owner rewrite is neither required nor justified by the receipt.
* Each of 12 cells constructs two independent natural banks and two reference fronts per bank: 48 front constructions. They repeat some common QKV/cache work, but factoring the front would require a separate reference-dataflow proof and is **not mixed into this epoch**.
* `_AB_DEQ` already caches each local expert's FP32 decoded W13/W2; there is no new dequantization-cache proposal here.
* Each of 72 independent model validations computes the back reference twice, once for the BF16 front and once for the FP8 envelope. That is 144 reference-back evaluations, including repeated pooler/expert/unpool work for the same immutable front and exactly equal actual returned routing IDs.

## One bounded implementation change

`tools/g_common_release_budget.py` reconstructs and asserts the frozen old all12 source first. It preserves all native strings, timed `kernel_fn` ASTs, reference arithmetic, routing/cache/adaptive tolerance checks, ownership/capture/guard handling and legitimate Marlin fallback. Only the qualification reference wrapper, preparation orchestration and budget diagnostics change.

The new per-cell memo retains at most one entry per internally owned reference front, at most four entries total. A hit requires the **same front and weights owners**, unchanged front tensor/version signature, and equal shape/dtype/device plus `torch.equal` over a **cloned full actual ID tensor**. Changed IDs force a fresh computation; no top-k substitution or assumed routing stability is allowed. Distinct banks cannot share a hit. Cached results are never handed to mutating functions in the retained oracle; unexpected Torch mutation of a cached result or saved ID snapshot fails closed. Finite-reference checks run on every use, and every output/routing/cache/finite/adaptive check still runs independently. Ordinary Torch front mutations invalidate the entry; native weight immutability remains the existing audited contract, as it already does for `_AB_DEQ`.

Cache entries clear at every cell end and unconditionally on preparation exit, including exceptions. No reference cache is used by the normal timed kernel or survives in returned production state. With identical actual IDs for each bank, this removes 96 duplicate back computations: 144 → 48. This is an operation count, **not a measured setup speedup or guarantee to meet 295 seconds**. If IDs vary, recomputation is mandatory and savings decrease.

Flushed `GQUAL_PHASE` markers record elapsed host time and current token/context around original prepare, clone/resources, base/fresh fronts, empty setup and each validation. A deadline failure prints `GQUAL_BUDGET_CONTEXT` with the last phase/cell. There are no markers or new operations inside model graph capture, no added comparative timing, no path selection and no deadline relaxation.

## Pins and validation

| Fixed arm | New bytes | New SHA256 |
|---|---:|---|
| common reference control (not an additional performance candidate) | 917408 | `42d8c0e4d26e52d195f5fc8d7802965ab73060be174f2315b5f2ebde8c3abc7b` |
| fixed short-RMW candidate | 917896 | `82234b757e64242fa23d704a909c6996c602e3f0dae327fb66a67260243544d8` |

Old common `4e3895be` and failed RMW `97935e7d` are not overwritten. The builder structurally compares every original top-level declaration except four preparation/reference-qualification functions; native literals, oracle arithmetic, thresholds and all timed entries are exact. New manifests are in the adjacent evidence directory. There are no new CUDA/C++ changes requiring a different native compilation claim.

Eleven actual Torch CPU tests pass in **10.465 seconds**, using the existing explicitly simulated CUDA/native arithmetic facade. Actual all12 orchestration preserves **72 model checks / 12 model captures / 36 changed replay checks** and **60 empty checks / 12 empty captures / 48 restored occupied checks** for both fixed arms. The fixture observes 48 back computations and 96 hits. Tests include changed live IDs, saved-ID independence, distinct banks, changed front, cached-result mutation/nonfinite rejection, exception cleanup, stale model/empty capture, invalid native metadata/flags/resources, output/reference NaN, cache corruption, guard/owner alias, fatal CUDA stand-in, budget and real cyclic Marlin-adapter fallback. This is source/host validation, not Hopper arithmetic, setup/capture or duration evidence.

Reproduce from this worktree's `tools`, with `OKC_REPO_ROOT` pointing at the tracked root checkout:

```text
/root/.cache/okc-host-validate/bin/python -m unittest test_g_common_release_budget -v
python tools/g_common_release_budget.py ROOT NEW_UNIQUE_OUTPUT --arm short-rmw
```

## Decision and stop criteria

Root alone owns any new official action. Keep the 295-second deadline and every original coverage/precision/progress/resource guard. Require peer review of the exact cache semantics/pin before consideration. A future successful runtime must complete all original counts, all13 correctness tests and all12 actual raw/adjusted benchmarks before interpreting a score. A setup pass alone is not a speedup.

If the new phase evidence shows original prepare/front creation already consumes the budget before reusable back computations, this mechanism has not addressed the bottleneck: stop rather than claim success or remove checks. Further front factoring or preparation-policy change would need a separately scoped source proof and explicit authorization. No public/GPU/API operation was performed for this implementation.
