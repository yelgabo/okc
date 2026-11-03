# Reset diagnostic owner-clone repair and operational audit

The old full-stage, short-path and four-cell payload pins are invalid for dispatch: their shared private-state clone rejects the actual `set` owners created by G preparation. This is a host correctness fix, with no native CUDA, math, fence, routing, timing policy or queue-order change. Root must review and replace the pins before resetting the existing queue.

Evidence baseline: main `65f8810`; original helper imported identically in child-only commit `2beac83` (root must **skip** that preparatory commit). The actual old helper and both whole preparation functions reproduce `RuntimeError: unreviewed state owner set` in the CPU facade. `g1b4_control_release.py` initializes `compose_active=set()` and nested `bfp.disabled=set()`; the latter receives scalar tuple policy keys from `bfp__route_key` and its skinny-alt branch.

## Minimal authorized change

Only `tools/dl_g1_layer_owners.py::_wl_clone` changes: clone exact builtin sets into a fresh memoized owner; allow only exact str/int/float/bool/None and recursively checked exact tuples as members. Tensor members (including hashable tensors inside tuples), arbitrary objects, frozensets and custom set/member subclasses remain unsupported. Tuple-member recursion is checked and opaque or mutable members reject; no general cycle acceptance is introduced. Repeated references to one set retain their alias within an owner, while separate clones get separate sets. Immutable tuple members may be shared. No change to Lt descriptors, immutable module sharing, tensor-storage alias reconstruction or scratch guards.

## New reproducible payload pins

`tools/reset_payload_regenerate.py` builds from tracked root generators twice. `--owners` permits reviewing this one helper delta without changing root. The verify package embeds the same owner helper through `dl_selected_pair.py:60`, and calls it in `dl_verify_pair_body.py:95`; it is affected too. Each new payload adds exactly 883 UTF-8 bytes. Full dependency and receipt pins are in [regeneration.json](evidence/2026-11-01-reset-host-readiness/regeneration.json).

| Diagnostic | Old pin / bytes | New SHA256 / bytes |
|---|---|---|
| Full six-load stage | d32d3c8a… / 906939 | aa49352105b39538b79c454f49e9f8e2014fa99b5c8205f145131169f18358e4 / 907822 |
| Short natural-route | 8308dcf3… / 903137 | 4a1a68f5824bd51c130e03eb736a2e1fb19e011735169400b0be2c0d17448183 / 904020 |
| Four-cell verify | 2bad52ce… / 968389 | 195bba34415dd614733100cc4e26022fd25891fad747ee393aaac9ab4c8d4fa0 / 969272 |

Full-stage proof remains owning job `74a29e0384324952`, native/empty PASS13, candidate `1607293eee3403109df5d37b6e2d1ee34b528c3578afa6c2c8668440afde1658`, evaluator `2999d8a201bb2b0bad6c7b3998af2143a1ed720d531dbb4d15670721ee9f957b`. That prior proof does not qualify the newly generated diagnostics' setup/runtime.

## Source-derived owner audit and CPU coverage

[Source inventory](evidence/2026-11-01-reset-host-readiness/source-owners.json) records exact assignment sites/expressions in the frozen G control, including conditional paths. It is **not an H200 state dump** or a claim that every conditional field exists. Native-eligible fixture types are derived from these sites:

- Packed weights, norms, router matrices, RoPE and graft coefficients are tensors; `ma_sfx` is a tuple of compiler-flag strings. `ma_w`/scratch are tensor containers. Policy caches are scalar/tuple-keyed dictionaries, with the two builtin sets above.
- `bfp.raw/streams/tabs/base4/sk2` contain immutable packed tensors/tuples; counters, fused partials and workspaces contain writable tensors/tuples. `comp`, `choice`, `sm90`, `old_paths`, diagnostics and disabled keys are builtin container/scalar values. Here `streams` are **packed weight tensors**, not CUDA stream objects.
- Actual source `_RouterExt` is compiled-code-only and shared. Actual `_GraftPlan` is a dict subclass with `lt/layouts/algo/weight/n/fp32`; clone retains descriptor/algorithm handles while cloning writable Lt workspace. Actual `_GraftLt` class is used via `__new__` in the fixture: native library/handle initialization is explicitly skipped. Read-only weight storage is shared and mutable workspace storage is disjoint.
- Fallback Marlin adapter objects are outside this native-only qualification, whose actual source checks native eligibility and, for stage, absent unused bank. This audit does not expand fallback support or claim opaque owners are safe.

Six real Torch 2.10 CPU tests pass. They execute the actual complete `_wl_prepare` and `_sp_prepare` control flow, actual clone/read-only/storage checks, fresh copying, summary/transport, module installation or original-state return, and prepare wrapper TF32/cache restoration. Stage completes 36 rows, 144 ordinary +144 fresh final validations, 12 empty calls; short completes 48 rows and 96 validations. Reports compress below the actual 11KB gates. Old-owner negative controls fail both preparations; set alias/member negatives, stale stage replay and both deadline rejection branches are exercised.

**Explicit stand-ins:** production native preparation is represented by the source-derived state fixture; reference/native arithmetic, CUDA capture/replay, events, memory budget and profiler are CPU substitutes. The actual `_wl_capture`/`_sp_graph` CUDA boundaries are replaced; the full orchestration functions are not. Synthetic timing is solely control-flow input and supplies no performance evidence. These tests catch the observed set defect and host owner/coverage/transport regressions; they do not establish actual preparation state completeness, CUDA graphs, kernel math, admission, profiling correctness or H200 latency.

Reproduce from this worktree (or root after cherry-pick):

```text
OKC_REPO_ROOT=<root checkout> /root/.cache/okc-host-validate/bin/python -m unittest reset_host_facade_test -v
# Run from tools; env override is optional after integration.
python tools/reset_payload_regenerate.py --root <root> --output <scratch> --owners tools/dl_g1_layer_owners.py
python tools/reset_owner_inventory.py <root> <output.json>
```

## Remaining operational risks and stop criteria

1. **Blocking before merge:** independently review the shared helper and all three new pins; regenerate from integrated tracked sources and require byte equality. No old frozen pin may be called ready.
2. **Setup deadline remains unqualified:** full-stage `_wl_prepare` actually uses **300s**, not 295s. Its guard is at case boundaries, not within the 24x4 timing /8x4 fresh replay loops, and cannot interrupt native preparation/builds. The short diagnostic uses 295s with inner timing/profiler guards, but 144 profiler contexts, graph work and base preparation have no measured combined H200 setup cost. Do not silently change either policy or add historical setup times as a forecast.
3. Stage hashes the entire weight bank in 8MiB CPU transfers, builds two module families and retains base+four clones/reference banks; source preflight must pass actual free memory/admission. CPU memory stand-ins do not validate this cost.
4. Actual H200 profiling events and rich natural-route report transport may differ from CPU substitutes. Require complete rows/checks/flags/resources and retained report, own job ID and correct board; incomplete or timeout means no causal/performance conclusion and no automatic retry.
5. Short trace intentionally fixes each natural input during timing and has no new fresh-input qualification; a future production change needs its own fresh graph/math gates. No production candidate or speed claim follows from this audit.

First two reset jobs and their order are unchanged in intent. No GPU, submit, automation or public action was performed. Parent owns final review and queue changes.
