# Occupied-to-empty fixture: bounded source and CPU diagnosis

November 1, 2026 Pacific / November 2 UTC. Failing implementation **2686a7653d6cf2325e944f419cdf0f81e4b6308a**, upload **c4a5ee29bcfeca248c0fee638ed103e9f211f7a8cc32b86bd3bb642676bf45e3 / 923915B**, own private **16b4db27421d4714**. Terminal receipt/reader rejection/archive published first at **d756f298b7f52cba8c0b6baf46497b4ee0701d22** in [terminal report](2026-11-01-rmw-common-pair-terminal.md). This follow-up is SOURCE/CPU only, as specifically requested. No production/runtime support, generated upload, reader or manifest is edited; no GPU action occurs.

## Finding and decision boundary

**A concrete fixture-contract defect is established. Device-fault causality is not.** The occupied fixture fills all eight route slots for each token with expert ID0. The actual layer's top-k router selects distinct expert indices for each token. Its cyclic local remapping is injective for retained IDs; masking nonlocal experts to-1 can repeat the empty sentinel, but cannot create repeated nonnegative local IDs. Thus a local expert can appear at most once per token, or at most T times across T tokens. This is the reachable production input contract of this internal native call, not an assertion that its low-level ABI explicitly validates uniqueness for arbitrary callers.

The frozen fixture is outside that domain for **both T1 and T8**, regardless of OFF/ON. At T1 it repeats one expert8times although the legal per-expert capacity is1. At T8 it repeats that expert64times although the legal per-expert capacity is8. This invalid input is introduced by the test, not by the production router or the RMW treatment.

The selected T8 source's fat-unit trap predicate is true under that64-entry routing count. It remains unproven that this trap was the instruction that produced CUDA719. The owning trace has no T/arm/replay ordinal, helper/compute instruction or device localization, and invalid routing can affect other paths before the trap. The native-count CPU facade intentionally models only counts and the source predicate, not GPU execution, scheduling or CUDA error delivery. No RMW performance or legal-workload production failure is inferred.

## Exact selected native predicate

The actual source factory capture selects Tiny for T1 and Tiny2 for T8. Mutable OFF0/OFF1 reuse the respective OFF module; ON0/ON1 reuse the corresponding ON module. All four selected code variants use **OKC_TINY=1**, default **OKC_SPLIT=0**, default **OKC_WGS=0**, **SK_T=8**, **FT_T=64**. The ON flag differs only by the registered SHORT_FINAL_RELAXED treatment; it does not change routing counts or fat classification. Exact flags/module names and shared native CUDA digest **3a91956f9d4a577cc3456baffcf02d9dacb82c875d6b0ff2deedf71bce4a7cf4** are recorded in [CPU evidence](evidence/2026-11-01-rmw-common-pair/occupied-contract-cpu.json) and the original [factory capture](evidence/2026-11-01-rmw-common-pair/factory-capture.json).

Native routing counts every valid top-k slot. With SPLIT=0, the selected source computes:

~~~cpp
const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
~~~

The prefix sum of nfj becomes S.njF. The non-WGS compute path invokes run_phase for A and B. There nF=S.njF*ubs, and under OKC_TINY the source contains if(nF>0)__trap(). For these shapes HD=ID=4096, so ubs is32 in phaseA and16 in phaseB.

| Input / selected module (OFF and ON) | Maximum entries for one expert | nfj for expert0 | Tiny fat trap predicate |
| --- | ---: | ---: | --- |
| Frozen T1 / Tiny |8|0|false from this count alone; input still violates distinct routing|
| Frozen T8 / Tiny2 |64|1|true: phaseA nF32; phaseB nF16 if reached|
| Legal repeated distinct IDs0..7, T1 / Tiny |1|0|false|
| Legal repeated distinct IDs0..7, T8 / Tiny2 |8|0|false|

If the modeled count-only sequence reaches the first T8 placement, the first offending arm is **OFF0**, because the function loops T1 then T8, and arms0,1,2,3. That is a source/model prediction, **not the observed failing GPU arm**. The source offers no reason to attribute this defect specifically to ON.

Relevant inspected source: runtime fixture tools/g_common_rmw_pair_body.py:_srp_occupied_empty; generated upload:_fork_native_moe selects Tiny/Tiny2; captured native routing computes nfj and run_phase contains the trap. The production _graft_route PACKED=0 loop repeatedly selects argmax then masks that exact expert to negative infinity; its packed/sorted alternatives also choose unique source expert positions. The selected short route dispatch uses PACKED=0. Per-token distinctness is preserved by lid=(ids-local_start+256)%256 and the lid<32 filter. No claim about NaN or otherwise invalid production inputs is needed for this domain argument.

## Smallest legal fixture proposal

Keep all current loads, arms, states, graphs, score weights, occupancy transitions, poison writes, empty checks and scratch-lifetime rules. Use one fixed distinct occupied pattern per token, with the eight local experts0..7 and unchanged normalized weights0.125. Keep the captured ids storage stable. The two local expression changes are:

~~~python
occupied_ids = torch.arange(8, device=dev, dtype=torch.int32).expand(t, 8).contiguous()
ids = occupied_ids.clone()
# In each existing occupied/empty transition:
ids.copy_(occupied_ids) if occupied else ids.fill_(-1)
~~~

Changing initialization alone is insufficient: the existing ids.fill_(0) in the transition loop would reintroduce duplicates before the first occupied replay. No scratch/counter reset, relaxed capacity, new kernel variant or route mask is needed. The proposal is evaluated only as an in-memory replacement of the frozen function in the diagnostic test; it is not installed into runtime support or an upload.

The pattern is deliberately maximally concentrated **within the legal domain** for T8: all8tokens choose the same8distinct experts, giving8entries to each. It continues to exercise occupied -> empty -> occupied -> empty on the same captured pointers for both loads and all four placements, with16zero-output empty checks. Local empty IDs-1 remain valid as the masked form of8global top-k experts outside this local shard.

## CPU regression and limits

[Diagnostic test](../../tools/test_g_common_rmw_pair_occupied_contract.py) extracts and executes the exact frozen fixture after verifying its SHA256, and captures the actual native source factories without compiling or accessing a GPU. It uses real CPU Torch tensors with explicit graph/native facades; these are contract/count/output stand-ins and do not test native arithmetic, barriers or CUDA graph correctness.

**5/5 tests PASS**. [Exact output](evidence/2026-11-01-rmw-common-pair/occupied-contract-tests.txt), [machine-readable observations/flags/proposed replacements](evidence/2026-11-01-rmw-common-pair/occupied-contract-cpu.json).

1. Actual factory flags/source select Tiny/SPLIT0/WGS0 and the stated fat predicate for T1/T8 OFF/ON.
2. A distinct-ID contract observer rejects the frozen fixture immediately at T1/OFF0.
3. A count-only native predicate observer allows the T1counts but rejects T8/OFF0 at64entries, reproducing the source predicate rather than CUDA719.
4. The prior permissive arithmetic-style mock passes all16empty checks despite duplicate IDs and a64-entry expert. This explains why the existing CPU coverage did not catch this input-domain defect: tools/test_g_common_rmw_pair.py:native_run checks only whether any ID is nonnegative before fabricating output.
5. The in-memory legal proposal passes all16empty checks and all40capture/replay observations, with unique retained IDs and per-expert count<=T for every arm. Each arm/load observes occupied capture followed by occupied, empty, occupied, empty replays; counts are T,T,0,T,0. An initialization-only partial repair is rejected.

The five tests ran in approximately0.03s after imports; the shell process took approximately2.3s. These tests make the proposed fixture/domain correction concrete and distinguish it from a permissive mock or partial repair. They do not prove that a repaired H200 job will pass, that resources/cleanup complete, or that preparation fits295s.

Reproduction from the registered source checkout, after generating the unchanged failing artifact with tools/g_common_rmw_pair.py ROOT .build/rmw-pair-v3/check.py:

~~~text
/root/.cache/okc-host-validate/bin/python tools/test_g_common_rmw_pair_occupied_contract.py docs/steering/evidence/2026-11-01-rmw-common-pair/occupied-contract-cpu.json
~~~

## Cost estimate, remaining risk and allocation choice

If captain chooses the narrow repair, estimated implementation/CPU/pin work is **15-25minutes**: apply the two fixture-local changes, add the distinct-route/capacity observer to the existing actual-entry-point CPU native facade, preserve the original negative regression, run the existing paired/source/reader CPU suite, and generate a separately named upload/manifest while retaining failing c4a5ee29. Verify native source and compiler flags byte-for-byte unchanged; new native compilation is not intrinsically required for a host-fixture-only change when those identities match. This is an engineering estimate, not a fresh check readiness claim or guarantee.

GPU work would still require a separately explicit allocation. Estimated incremental metadata work is one tiny occupied-ID tensor and fixed-pattern copies in place of fill0; there are no extra graph replays, native modules, oracle calls or matched timing rounds. Its setup-time effect and the whole paired295s fit remain unmeasured. The failed237.581sworker duration includes more than preparation and cannot serve as that bound. A repaired job could still fail for another defect or cost reason.

**Recommendation to captain:** the discovered input-domain defect makes one narrow fixture repair technically justified if its15-25minute cost and another private slot beat the researcher's strongest alternative. There is no evidence here for an RMW speedup or for automatically consuming the heldout reserve. Captain must explicitly choose repair plus reallocation versus stopping RMW; no work beyond this source/CPU finding or GPU dispatch is inferred. The existing all-six/heldout/duplicate-null/regression promotion thresholds remain fixed. README/index remains captain-owned.
