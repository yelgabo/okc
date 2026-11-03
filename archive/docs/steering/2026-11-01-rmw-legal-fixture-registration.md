# Legal occupied fixture: replacement paired private check registration

November 1, 2026 Pacific / November 2 UTC. Explicit captain reallocation **439f9b3** supersedes the original heldout#3 reservation and authorizes **one replacement matched private check as tranche#3/4** after these exact CPU/source/pin gates and fresh quota/no-live preflight. Terminal predecessor **16b4db27421d4714**, archived **d756f29**; bounded diagnosis **52628b6**. README/index remains captain-owned. This document is published before registration; append exact acceptance/outcome below.

## Ready pins and exact delta

- Upload **.build/rmw-pair-v4/check.py**, **924005B**, SHA256 **785a90ca08ddde0c3b194fc219a0645cb8cd14bac02620d8714f0b66163a0b66**.
- tools/g_common_rmw_pair_manifest.json: **4821B**, SHA256 **c5f0e3975e30f4c012be8320615c7a19bbf9a7b7a63381906c10f3a3b3d4c574**.
- Reader tools/g_common_rmw_pair_read.py unchanged: **10920B**, SHA256 **745d7bb508d170dbc6bcd4b86e0f8a0b39c9a9ba08885aec54a45a33e68f004d**. Its sibling manifest binds this new upload; both old/new readers reject the other's source.
- Device prefix unchanged: **aea9735e7d91a7d51a7afe63c8cf0774b5bbd15de6ab416bbc5acbd2469ddbd2**. New harness core **e5229e040d90ddb2f27ef1072f49d346aac87c6c123ef66f2ea4124e9f2f732f**.
- Judged repaired-common OFF **4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280**; raw ON **2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd**, unchanged.
- Failing upload preserved separately at .build/rmw-pair-v3/check.py, **c4a5ee29bcfeca248c0fee638ed103e9f211f7a8cc32b86bd3bb642676bf45e3 / 923915B**. Its exact frozen reader/manifest are copied under [v3-frozen-reader](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/v3-frozen-reader) and retain original source rejection/identity semantics.

Only two fixture-local expressions change: occupied_ids is arange8 expanded over tokens and ids is its clone; every occupied transition uses ids.copy_(occupied_ids), while empty transitions retain fill_(-1). Captured ids allocation/storage is not rebound. Both loads/allfour mutable placements retain the same occupied -> empty -> occupied -> empty sequence, captured output poison and **16empty checks**. No scratch/counter reset or native kernel change occurs.

The complete generated diff has only that function and the derived _WL_CORE_SHA pin change. The production/device prefix is byte-identical; every other AST node, including production and timed functions, arm math, measurement loop, cleanup, oracle, memo, ownership, graph and resource logic, is identical. All **seven actual native source factories**, including CUDA/C++ source, names, compiler flags and exports, compare exactly equal to v3. Prior selected local12.8 compiler evidence therefore remains applicable to those unchanged native inputs; no new compilation or GPU qualification is claimed.

## CPU/source gates

**23/23 CPU tests PASS**, including the preserved paired/source/owner/reader suite, historical builder/reader regressions, five bounded diagnostic cases, and two new generated prepare-entry negatives. The CPU native facade now independently enforces per-expert count<=T and per-token distinct nonnegative IDs. Full generated prepare rejects the frozen duplicate fixture at the capacity guard; a distinct synthetic below-capacity duplicate is rejected at the unique-ID guard. Both failures retain cleanup tests. Corrected full prepare passes in both caller inference contexts and preserves36rows/144math/144fresh/32empty/16occupied-to-empty/432attention witnesses and the original ownership/resource/reader negatives under explicit CPU native/math/CUDA stand-ins.

Exact deterministic regeneration and source/reader binding pass. The old and new sources remain separately hash-verified; reader code is unchanged and each manifest rejects the other source. Native/timed source invariance is source evidence, not H200 correctness, resource admission or measured fit.

Evidence: [CPU receipt](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/cpu-tests.json), [source gates](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/source-gates.json), [exact upload delta](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/source-delta.patch), [factory equality](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/factory-invariance.json), [file pins](../../../docs/steering/evidence/2026-11-01-rmw-common-pair-v4/file-pins.json).

## Frozen experiment and stops

Hypothesis: the registered Tiny/Tiny2 final-counter RMW representation yields a reproducible whole-six raw cold-graph gain over CURRENT repaired-common OFF under matched inputs/routing and independently prepared mutable placements. This is the same question as v3; the only repaired instrument input now respects the actual distinct-topk domain. Both arms and official OFF retain every common progress/cooperative/preexpert fix. No historical607/unrepaired comparator is run.

Keep original **993811/993817/993821**, all **36rows**, **4placements/two code variants**, **24cold randomized rounds**, **256MiBflush**, six loads **1/8/16/32/128/256**, **46:16** weighting, and allfour OFF/ON crosses. Promotion still requires every bank/cross whole-six raw gain to exceed **log1.01** and **2xduplicate log-null**, with each context/load regression **<=max(0.5us,1%)**. Oracle/fresh/empty/resource/actual-exercise/private-ownership/cleanup/own-terminal-success gates are unchanged. Official OFF score is descriptive; the registered matched private comparison answers attribution within its scope.

The **295s total guard** includes normal preparation, private exercise and cleanup. Fit remains plausible but unmeasured: prior c193 normal prepare106.901449s leaves188.098551s at that observed cost; native build set is unchanged from v3. Tiny metadata-pattern setup/copy cost is unmeasured. Failed v3's237.581sworker duration is not prepare timing, a budget bound or proof that this repair alone resolves its CUDA719. No localized719 cause is claimed.

**Any further setup/instrument failure stops this RMW allocation; no automatic#4retry. A valid null/loss/subthreshold/inconsistent-bank/regression failure stops RMW-only promotion.** #4 is reserved for fresh compilation/preparation heldout confirmation **only if#3 validly passes**, using previously frozen **202611021/202611023/202611029**. Do not repeat the exact ON single-arm qualification already completed by c193. Public/scored submission remains unauthorized.

## Dispatch ownership and next gate

Implementation thread is sole dispatcher. Publish these pins/results, refresh authenticated private quota and latest layer jobs, require positive private quota/no live layer job, regenerate/verify this exact upload again, and call check once with no-wait JSON. Wait only on its accepted own ID and confirm board/mode/source association. Next gate is terminal official success plus frozen reader/capsule evidence, or a specific setup failure stop. No additional approval cycle is required after these authorized conditions clear. Captain owns campaign/index updates; no README edit is included here.


## Accepted replacement private #3 — November 2, 02:22:46 UTC

Preregistration/implementation **d5f29167b34f78db864d22550b23f4a0516f6a21** was pushed before dispatch; all11 changed remote file identities were verified. Captain independently confirmed physical upload/manifest and the narrow delta. Fresh authenticated preflight showed **8private/5scored remaining**, with **no live layer job among latest50**. Deterministic regeneration and the physical785a90ca/924005B upload matched immediately before the single check call.

**Own accepted ID c5e4f89672f743bb**, board muse_decode_layer, mode check, team yelgabo, created **2026-11-02T02:22:46.573Z**. First owning status is **queued/result null**. Server content hash **28d360741906efb0ff5a90be98b92a053aff04480442d92992d47907ee557cab** and payload reference match the owning receipt; that server identity is distinct from the physical SHA256. Immediate postaccept quota explicitly reports **7private/5scored remaining**, used3checks. **3/4 initial tranche slots now have accepted IDs.**

Exact-own-ID wait is active in exec session **11629**, with script digest/command archived in wait-registration.json. It preserves raw terminal stdout before parsing. One new private check was issued; no public/scored action, duplicate registration or additional experiment. Source/reader/manifest are frozen for this accepted job. There is no startup, correctness, resource, paired timing, official score or promotion result yet. Next gate is this same job's owning terminal/frozen reader/archive decision. Further instrument failure stops the RMW lane; valid null/loss/subthreshold stops RMW-only promotion; #4 remains fresh heldout only after a valid #3 gate pass. README/index remains captain-owned.


## Terminal supersession — November 2, 02:48:30 UTC

Own c5e4f89672f743bb is done/PASS; existing wait11629 completed. Frozen unchanged reader accepts complete instrument evidence and returns private_raw_gate=false. All3banks/all12crosses fail the1%/2xnull gain gates, bank993821 also fails one context regression. RMW-only promotion is closed; conditional#4heldout is ineligible and remains unused. Official commonOFF2.112499365 is descriptive only. [Complete terminal report and exact archive](2026-11-01-rmw-legal-fixture-terminal.md) supersede the queued snapshot above. No remeasurement, public/scored action or source tweak. Captain retains README/index.
